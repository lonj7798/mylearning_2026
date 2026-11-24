<!-- chapter: ch-45d
     track: rl
     kind: content
     title: Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL
     deps: [ch-45c, ch-35]
     sources: [[kimi-k2]], [[kimi-k2-recipe]], [[kimi-k2-agentic-data]], [[kimi-k2-5]], [[kimi-k3]], [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[deepseek-v3.1]], [[deepseek-v3.1-recipe]], [[qwen3-coder]], [[qwen3-coder-next]], [[minimax-m2-aligning-to-what]], [[minimax-forge-agent-rl]], [[nemotron-3-super]], [[tongyi-deepresearch]], [[deepswe]], [[deepswe-recipe]], [[toucan-mcp]], [[agent-data-protocol]], [[swe-smith]], [[r2e-gym]], [[swe-gym]], [[grpo]], [[rlvr-beyond-base-model]]
     figures: figures/agentic-stage-map.html
     revised: 2026-09 (generality revision)
-->

# 제45d장 — 공개 agentic recipe를 나란히 비교하기: stage 배치, 데이터 혼합, agentic RL

> **핵심 통찰.** 공개 agentic recipe는 거의 모든 hyperparameter에서 의견이 다르지만, 세 가지 배치에는 동의한다. 첫째, agentic trajectory는 **pre-training**이 아니라 **mid-training**에서 base model에 들어간다. [[glm-4-5]]는 pre-training 22T token 뒤 128K에서 agent trajectory 100B token을 넣고(§2.3, Figure 3), [[glm-5]]는 28.5T budget 안에 1.55T token의 32K/128K/200K mid-training block을 둔다(§2.3). [[qwen3-coder-next]]는 여기에 repository-level token 약 600B를 두고(§3.1.1), [[nemotron-3-super]]는 25T-token pre-training data에 agentic corpus를 아예 열거하지 않는다(§2.3). 둘째, report가 늘리는 것은 RL algorithm이 아니라 **RL environment 수**다. [[deepseek-v3.1]]는 합성 general-agent environment 1,827개(§3.2.3)와 agent prompt 85,267개(Table 1의 네 task count 합)를 만들고, [[glm-5]]는 9개 언어에 걸친 검증 가능한 software-engineering environment 1만 개 이상을 쓴다(§4.2.1). [[qwen3-coder-next]]는 검증 가능한 task instance 약 800K개를 쓰며(§2.1), [[tongyi-deepresearch]]는 agentic RL이 “사용하는 특정 algorithm보다 data 품질과 training environment 안정성에 더 의존한다”고 명시한다(§3.4.3). 셋째, **한 agent harness에서 훈련한 model은 다른 harness로 전이되지 않는다.** [[qwen3-coder-next]]는 제한된 cross-scaffold transfer를 측정하고(§3.1.2, Figure 3), [[minimax-m2-aligning-to-what]]는 scaffold를 바꾸면 benchmark score가 “plummet”한다고 보고한다. [[kimi-k3]]는 RL 동안 harness를 무작위화하여 이에 대응한다(§4.2.1).
>
> **지침.** agentic capability가 general capability와 공존해야 하면, environment마다 single-environment stage를 순서대로 실행하지 말고 모든 environment 위의 mixed RL stage 하나로 훈련한다. [[nemotron-3-super]]는 “모든 environment를 동시에 훈련하면 안정된 gain을 얻지만, single-environment training은 다른 benchmark에서 심각한 regression을 낳는다”고 보고한다(§3.2.1). [[agent-data-protocol]]은 SFT에서도 같은 현상을 잰다. 13-dataset mixed corpus는 7B에서 SWE-Bench Verified 10.4%에 도달하지만 SWE-smith-only tuning은 1.0%이고, GAIA에서는 9.1% 대 AgentInstruct-only 0.6%였다(§6.2, Table 6). throughput 때문에 stage를 분리해야 할 경우, 예컨대 긴 software-engineering rollout을 short-horizon environment와 같이 학습하면 batch가 굶는 경우에는 분리한 다음 다른 stage를 회복해야 한다. 명시적 distillation([[glm-5]] §3.5, [[kimi-k3]] §4.1.3)을 하거나, 분리한 stage 뒤 general objective를 다시 포괄하는 reward stage를 둔다. [[nemotron-3-super]]는 throughput 때문에 SWE-RL을 분리한 뒤(§3.2.2), multi-environment RL에서 사용한 것과 같은 generative reward model로 채점하는 RLHF stage를 실행한다(§3.2.3). 훈련에 harness 하나만 쓸 수 있다면 결과 model은 harness-specific일 것이라고 명시해야 한다. 여러 harness에서 훈련하지 않고 scaffold 전반으로 일반화하는 recipe를 보고한 source는 없다.

## 범용 모델에서 이 장이 중요한 이유

앞 장들은 agentic mid-training data(ch-32d), multi-skill SFT mixture(ch-30b), environment와 task synthesis(ch-29c), multi-turn agentic RL mechanics(ch-45b), context management(ch-45c)처럼 stage 하나씩을 분리해 다뤘다. 이 장은 그 stage 전체에 걸친 공개 recipe 11개를 한 번에 읽고, single-stage 장이 답할 수 없는 질문을 묻는다. 고정 budget에서 agentic capability는 어디에서 구축해야 하며, 각 배치는 general capability에 어떤 비용을 내는가?

agentic recipe는 pipeline의 모든 stage를 건드린다. pre-training → mid-training → SFT → preference optimization → RL → evaluation이다. 이것이 generality 위험의 근원이다. 한 harness 위의 늦은 RL stage 하나에서 agentic ability를 더한 model은 broad capability 위에 narrow competence를 추가한 것이며, 여기서 읽은 11개 report 중 agentic stage 전후에 non-agentic evaluation을 실행한 것은 3개뿐이다(§7: [[kimi-k2-5]] Table 2, [[qwen3-coder-next]] §5.3, [[nemotron-3-super]] §3.2.1--§3.3).

[figures/agentic-stage-map.html](figures/agentic-stage-map.html)의 interactive cross-lab map에는 locus와 함께 §1의 모든 cell이 있다. §1--§4를 읽을 때 보고서별로 어느 cell을 비워 두었는지 확인하는 데 사용한다.

## §1 recipe가 agentic capability를 둘 수 있는 여섯 위치

**정의.** *agentic stage placement*란 recipe가 model action(tool call, code, message)과 environment observation이 번갈아 나오는 full action-observation trajectory를 model에 처음 노출하는 training stage다. 가능한 여섯 stage는 pre-training, mid-training, SFT, specialist 또는 expert RL, main RL stage, 최종 distillation 또는 merge다.

**측정 가능한 문제.** 각 placement는 token, verification 노력, forgetting 위험에서 다른 가격으로 capability를 산다. Pre-training token은 token당 싸지만 검증할 수 없다. RL rollout은 검증할 수 있지만 sample당 비용이 몇 자릿수 더 크다. 모든 것을 RL에 두는 recipe는 capability 단위당 가장 높은 비용을 내며, 모든 것을 mid-training에 두는 recipe는 verification signal을 전혀 얻지 못한다.

**cross-lab map.** 11개 report를 각 locus에서 읽으면 다음과 같다. 빈 cell은 그 report가 agentic data에 대해 그 stage를 설명하지 않는다는 뜻이다. 어떤 진술이 없다는 사실은 그 stage를 건너뛰었다는 증거가 아니다. 명명 주의점 하나가 있다. library card [[deepseek-v3.1]]에는 **DeepSeek-V3.2** technical report(arXiv:2512.02556)가 들어 있다. 이전 장 link가 해석되도록 slug를 그대로 유지했다. 아래 해당 link의 모든 §-locus는 V3.2 report의 locus이며, Recipe 표는 checkpoint를 DeepSeek-V3.2로 표기한다.

| Report | Pre-train | Mid-train | SFT | Specialist RL | Main RL | Final distill |
|---|---|---|---|---|---|---|
| [[kimi-k2]] | 미보고 | anneal + long-context만(§2.5) | MCP + 합성 tool trajectory(§3.1.1) | 미보고 | joint RL stage 하나(§3.2) | 미보고 |
| [[kimi-k2-5]] | vision corpus의 GUI/agent trajectory(App. B.3) | long-context activation(§4.3) | zero-vision, text-only(§2.2) | RL 안의 ability-based domain(§2.3) | joint text-vision RL + PARL(§4.4.2, §3) | 미보고 |
| [[kimi-k3]] | 미보고 | cooldown 256K→1M(§3.4) | prior-Kimi expert의 cold start(§4.1.1) | 9 expert = 3 domain × 3 effort(§4.1.2) | partial rollout, budget reward −1(§4.1.2) | MOPD(§4.1.3) |
| [[glm-4-5]] | 미보고 | 128K에서 agent+long 100B(§2.3) | 4-step agent pipeline(§3.1) | Reasoning / Agent / General-chat(§3) | search + SWE agentic RL(§3.3) | 3 expert에 대한 unified SFT(§3.1) |
| [[glm-5]] | 미보고 | 32K 1T / 128K 500B / 200K 50B(§2.3) | agent share 확대, context 202,752(§3.1) | slide-generation expert만(§4.2.5) | Reasoning → Agentic → General RL(§3.2--§3.4) | on-policy cross-stage distillation(§3.5) |
| [[deepseek-v3.1]] | 미보고 | sparse-attention continued training(§2.1.1) | prompt-based cold start(§3.2.2) | agentic/reasoning 6개 + 기타 domain 2개(§3) | mixed RL stage 하나(§3) | specialist data를 RL 전 distill(§3) |
| [[qwen3-coder]] | 7.5T token, code 70%(blog) | 미보고 | 미보고 | 미보고 | code RL + agent RL, environment 20,000개(blog) | 미보고 |
| [[qwen3-coder-next]] | 미보고 | 262,144에서 repo token ~600B(§3.1.1) | execution-verified trajectory(§4.1) | WebDev / UX / QA / SWE(§4.2) | 검증 가능한 SWE task ~800K(§2.1) | expert를 SFT model로 distill(§4.2.5) |
| [[nemotron-3-super]] | 25T token, agentic corpus 열거 없음(§2.3) | 미보고 | 7M sample / 80B token(§3.1.2) | 미보고 | RLVR(21 environment) → SWE-RL → RLHF(§3.2) | 미보고 |
| [[tongyi-deepresearch]] | 미보고 | Agentic CPT 32K 뒤 128K(§3.3.1) | ReAct + context-management cold start(§3.4.2) | 미보고 | on-policy GRPO variant(§3.4.3) | model merging(§3.4.4) |
| [[minimax-forge-agent-rl]] | 미보고 | 미보고 | 미보고 | 미보고 | unified mixed-domain CISPO(§4.1) | 미보고 |

**map 읽기.** 세 column은 촘촘하고 세 column은 드물다. 11개 report 중 mid-training은 8개, SFT는 9개, main RL은 11개가 기술한다. Agentic content를 pre-training에 둔 report는 두 개뿐이고 둘 다 target한 배치가 아니라 incidental하다. [[qwen3-coder]]는 trajectory를 주장하지 않은 채 7.5T token의 code 비율을 70%로 올리고, [[kimi-k2-5]]는 GUI action trajectory를 vision data 7 category(caption, interleaving, OCR, knowledge, perception, video, agent data) 중 하나로 열거하지만 token share를 제시하지 않는다(App. B.3). Specialist RL과 final distillation은 [[glm-4-5]], [[glm-5]], [[deepseek-v3.1]], [[kimi-k3]], [[qwen3-coder-next]] 다섯 report가 함께 채우며, [[kimi-k2]], [[qwen3-coder]], [[nemotron-3-super]], [[minimax-forge-agent-rl]] 네 report는 둘 다 비워 둔다. 남은 두 report는 각각 한 column만 채운다. [[tongyi-deepresearch]]는 specialist를 훈련하지 않고 끝에서 model을 merge하고, [[kimi-k2-5]]는 RL 한 run 안에서 ability domain으로 정리하며 별도 specialist-distillation stage가 없다고 밝힌다(§2.3). 이 분할은 시간순이 아니다. 2025년 [[glm-4-5]] report도 이미 두 column을 모두 가진다.

**범용 모델에 주는 함의.** 지배적인 placement는 engineering 단위당 data volume이 가장 큰 mid-training과 SFT 두 단계, 그리고 verifier가 있는 한 RL stage다. 이 순서는 generality 논지와 일치한다. broad competence는 volume에서, correctness는 verification에서 오며, 둘 중 어느 것도 다른 하나를 대체하지 못한다.

## §2 Mid-training: token 수와 합성 데이터의 상한

**정의.** *Agentic mid-training*은 main pre-training run 뒤, SFT 전에 실행하는 continued pre-training이다. agent trajectory와 repository 규모 또는 long-document data에 next-token-prediction loss를 적용한다. [[tongyi-deepresearch]]는 이를 Agentic Continual Pre-training이라 부르며, 목적을 “agentic behavior에 대한 강한 inductive bias를 갖추면서 broad linguistic competence를 동시에 보존한 base model”을 제공하는 것이라고 설명한다(§3.3.1).

**report가 실행한 순서의 mechanism.**
1. Agent trajectory는 길기 때문에 context window를 단계적으로 확장한다. [[glm-4-5]]는 4K → 32K → 128K(§2.3), [[glm-5]]는 32K → 128K → 200K(§2.3), [[tongyi-deepresearch]]는 32K → 128K(§3.3.1), [[qwen3-coder-next]]는 32,768에서 262,144로 옮긴다(§3.1.1).
2. 자연 long-form data를 먼저 두고 합성 trajectory를 마지막에 둔다. [[glm-5]]는 더 뒤 stage에서 “Long documents and synthetic agent trajectories”를 up-sample한다(§2.3).
3. Base distribution을 유지하려고 전체 과정에 general pre-training data를 interleave한다. [[tongyi-deepresearch]]는 “a small proportion of general pre-training data is interleaved”라고 쓴다(§3.3.1).
4. Packing policy를 trajectory boundary에 맞춘다. [[glm-4-5]]는 mid-training에서만 best-fit packing을 적용하고(§2.3), [[qwen3-coder-next]]는 best-fit packing에 “highly repetitive segments” masking을 더한다(§3.2).

**수치 예시: budget 중 agentic 비율은 얼마인가?** Mid-training과 total token count를 둘 다 적은 두 report를 사용한다.

| Report | 전체 base token | Mid-training block | Agent trajectory를 포함한 stage | 전체 대비 비율 |
|---|---|---|---|---|
| [[glm-4-5]] | 23T(Abstract) | 500B + 500B + 100B = 1.1T(Figure 3) | 128K에서 100B | 100B / 23T = 0.43% |
| [[glm-5]] | 28.5T(§1, §2) | 1T + 500B + 50B = 1.55T(§2.3) | 128K와 200K에서 550B | 550B / 28.5T = 1.9% |

손으로 산술을 확인하면 `100 / 23,000 = 0.00435`이고 `(500 + 50) / 28,500 = 0.0193`이다. GLM-5는 두 release 사이 agentic-adjacent share를 약 4.4배 올렸고, total budget은 1.24배 커졌다. 따라서 GLM-4.5에서 GLM-5로의 agentic capability 증가는 total token budget만으로 설명되지 않는다.

**근거: 합성 데이터는 얼마나 많으면 과한가?** [[qwen3-coder-next]]는 상한과 이유를 모두 말하는 유일한 report다(§3.1).

```
합성 데이터에 크게 의존하면 target task 성능은 상당히 개선할 수 있지만,
과도한 특화, 응답 다양성 감소, fine-tuning 동안 다른 task로의 약한 적응으로
이어질 수 있다. 따라서 목표는 응답 다양성과 강한 범용 capability를 유지하면서,
model이 일반 user task를 안정적으로 수행하는 데 필요한 최소량의 합성 데이터를
도입하는 것이다.
```

해당 positive result는 synthesis가 아니라 rewriting에서 나온다. Web document를 Qwen3-Coder-480B-A35B-Instruct에 통과시켜 깨끗한 Markdown으로 만들면 Evalplus가 54.38에서 63.09, MultiplE가 36.02에서 48.35, CRUX-Eval이 57.13에서 58.94로 오른다(§3.1.1, Table 1). **결과(단일 연구)**다. 이 ablation의 base model, token count, seed count는 보고되지 않았다.

**조건과 한계.** 같은 total budget의 control run과 비교해 mid-training agentic share를 ablate한 report는 없다. 위 표의 모든 수치는 측정된 optimum이 아니라 recipe value다. 따라서 이것은 **Open question**이다. 잃는 general capability 1점당 agentic capability를 최대화하는 share는 어떤 공개 report도 측정하지 않았다.

## §3 SFT 비율: trajectory sourcing과 공개 dataset

**정의.** *SFT agentic share*는 supervised fine-tuning sample 중 single-turn response가 아니라 full action-observation trajectory인 sample의 비율이다.

**trajectory의 출처.** verification 강도가 낮은 순에서 높은 순으로 네 sourcing pattern이 나타난다.

1. **Simulated tool과 simulated user.** [[kimi-k2]]는 GitHub에서 실제 MCP tool 3,000개 이상을 가져와 합성 tool 20,000개 이상으로 evolve하고, 합성 system prompt로 agent 수천 개를 만든다. LLM-simulated user와 stateful tool simulator로 trajectory를 생성한 뒤 LLM judge가 task rubric을 충족한다고 표시한 trajectory만 유지한다(§3.1.1; [[kimi-k2-agentic-data]] 참조). [[glm-4-5]]도 같은 네 단계를 실행하며 multiple judge agent의 일치를 요구한다(§3.1).
2. **대규모 synthetic multi-role simulation.** [[nemotron-3-super]]는 6-stage pipeline을 실행한다. domain → policy와 tool → scenario → policy-scenario pair당 simulated interaction 16개 → outcome 및 process 수준 LM-as-a-Judge verification → all-success와 all-failure scenario를 버리는 difficulty filtering 순서다. 이로써 직전 Nano release의 5 domain 15,588 conversation과 비교해 “838 domain 전반 279,116 conversation”을 생산한다(§3.1.1).
3. **실제 harness 안의 teacher trajectory.** [[nemotron-3-super]]는 Codex, OpenCode, Qwen Code CLI, Stirrup 전반에서 Qwen3-Coder-480B와 MiniMax M2.5의 CLI interaction을 기록하고 OpenAI message format으로 정규화한다(§3.1.1). [[qwen3-coder-next]]는 여섯 scaffold 전반에서 Qwen3-Coder-480B-A35B-Instruct로 trajectory를 생성하고, termination signal 누락, task failure, malformed tool call을 rule-filter한다(§3.1.2).
4. **실행으로 검증한 trajectory.** [[qwen3-coder-next]]는 Mini-SWE-agent instance를 user simulator로 배치한다. 이 instance가 제안 code를 실행하고 compiler output, runtime error, environment state를 바탕으로 response가 task를 전진시키는지 판정한다(§4.1).

**수치 예시: 공개된 SFT blend 하나의 agentic share.** [[nemotron-3-super]]는 stage-1 SFT에 “over 7M total samples”가 있다고 쓰고(§3.1.2), agentic component 대부분의 크기를 공개한다. 공개된 agentic component를 합하면 다음과 같다.

```
general-purpose tool calling      1,500,000
conversational tool use             279,116
terminal use                         84,864
agentic CLI synthesis                15,000
SWE task                              3,000
web development                      10,000
                                 ----------
subtotal                          1,891,980   →  1.89M / 7M ≈ sample의 27%
```

report는 정확한 blend percentage를 출력하지 않는다(Figure 16에만 나타난다). 그러므로 27%는 공개된 component 크기에서 얻은 lower bound다. 크기가 출력되지 않은 search 및 long-context agentic data를 빼고 계산했다. 이 산술의 핵심은 규모다. SFT에서 agentic capability를 겨냥하는 recipe는 sample의 몇 %가 아니라 대략 4분의 1에서 3분의 1을 trajectory에 할당한다.

**합성 pipeline이 없는 독자를 위한 공개 dataset.** 공개된 세 set이 재현 가능한 비용으로 같은 범위를 다룬다.

| Dataset | Environment | 크기 | 해당 locus의 결과 |
|---|---|---|---|
| [[toucan-mcp]] | crawl한 약 2,800개 중 선별 MCP server 495개, tool 2,000개 이상 | trajectory 1.5M; SFT subset 119.3K | Qwen2.5-32B-Instruct BFCL V3 61.73% → 70.45%(+8.72); 14B 57.69% → 65.09%; 7B 55.10% → 58.26%(Table 2) |
| [[swe-smith]] | Python repository 128개, 각각 Docker image 하나 | task instance 50,137개; curated trajectory 5,016개 | SWE-agent-LM-32B가 SWE-bench Verified 40.2%, Lite 30.7%, pass@1(Table 3) |
| [[agent-data-protocol]] | coding, SWE, tool use, browsing을 아우르는 변환 dataset 13개 | trajectory 1.3M | SWE-Agent로 Qwen2.5-Coder-32B가 SWE-bench Verified 40.3%; base model 대비 평균 gain 약 20%(§6.1) |

**수치 예시: 실행 가능한 environment 비용.** [[swe-smith]]는 repository 128개 전반 instance 50,137개에 총 $1360, 저자 한 명의 노동 약 20시간, environment storage 295 GB를 보고한다(§2.2, Table 2). Instance당 `$1360 / 50,137 ≈ 2.7 cents`이고 `295 GB / 50,137 ≈ 5.9 MB`다. Table 1은 bug generation만의 instance당 평균 model cost 2.32 cents를 별도로 출력한다. 2.7-cent 수치는 repository installation과 issue writing까지 포함하는 전체 $1360을 나눈 값이다. 같은 표의 비교 행에서 R2E-Gym subset([[r2e-gym]])은 task 4.6k개에 4 TB, task당 약 870 MB이고, [[swe-gym]]은 task 2.4k개에 6 TB, task당 2.5 GB다. 차이는 instance당 image 하나가 아니라 repository당 image 하나를 두는 design choice다. Task당 5.9 MB 대 870 MB, 즉 task당 storage 약 147배다. 세 task count가 50,137, 4.6k, 2.4k로 다르므로 비교는 고정 total이 아니라 task당 기준에서 성립한다.

**SFT 수치와 함께 봐야 할 negative result.** [[deepswe]] §6은 자체 setting에서 SWE-Smith와 SWE-Gym data의 RL이 “high solve-none rate와 함께 limited improvement”를 냈다고 보고한다. SFT material로 효과적인 dataset이 자동으로 RL prompt pool에도 효과적인 것은 아니다. RL에는 현 policy에서 pass rate가 엄밀히 0과 1 사이인 prompt가 필요하기 때문이다.

## §4 Specialist distillation 대 하나의 mixed RL stage

**정의.** *Specialist distillation*은 common checkpoint에서 여러 domain-specific policy를 훈련한 뒤, specialist output으로 student를 훈련해 하나의 model로 합치는 방식이다. 대안인 *one mixed RL stage*는 모든 domain의 prompt를 domain별 reward function과 함께 하나의 RL run으로 보낸다.

**측정 가능한 문제.** 순차적 single-domain optimization은 앞 domain을 저하시킨다. [[glm-5]]는 “서로 다른 objective를 순차적으로 최적화하면 이전에 습득한 capability가 누적적으로 저하될 수 있다”고 직접 쓴다(§3.5). [[deepseek-v3.1]]는 같은 문제를 반대 방향에서 서술한다. domain을 RL stage 하나로 합쳐 “multi-stage training paradigm에 흔히 수반되는 catastrophic forgetting issue”를 피한다(§3).

**현재 report의 여섯 design.**

| Design | Report | 구조 |
|---|---|---|
| Specialist → distill → mixed RL 하나 | [[deepseek-v3.1]] | agentic/reasoning domain 6개와 writing·general QA를 각각 같은 base에서 RL 학습; 그 data를 하나의 model로 distill; 이후 mixed RL stage 하나(§3) |
| Expert → unified SFT | [[glm-4-5]] | Reasoning, Agent, General-chat expert 각각이 cold-start SFT 뒤 RL을 거치고, 수백만 sample의 SFT set으로 distill(§3, §3.1) |
| Sequential RL → on-policy distillation | [[glm-5]] | Reasoning RL → Agentic RL → General RL 뒤 earlier stage의 final checkpoint에서 distillation(§3.2--§3.5) |
| Expert × effort → MOPD | [[kimi-k3]] | domain 3개 × reasoning effort 3개 = expert 9개, Multi-Teacher On-Policy Distillation으로 통합(§4.1.2--§4.1.3) |
| Specialist 없음, staged RL | [[nemotron-3-super]] | 모델 하나가 RLVR → SWE-RL → RLHF → MTP healing을 차례로 거침(§3.2) |
| Specialist 없음, stage 하나 | [[minimax-forge-agent-rl]] | Reasoning, General QA, Agent를 CISPO 아래에서 동시에 혼합(§4.1) |

**두 lab이 쓴 on-policy distillation mechanism.** Student가 자기 trajectory를 sample하고, teacher가 student token에 score를 매기며, 그 ratio가 advantage를 대체한다. [[glm-5]](§3.5, Eq. 2)는 다음과 같다.

```
Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ]
```

[[kimi-k3]](§4.1.3, Eq. 15)는 clip을 더한다.

```
r_opd^d(y_t | e, x, y_<t) = clip( sg[ log( π_teacher^(d,e)(y_t | x, y_<t) / π_θ(y_t | e, x, y_<t) ) ],
                                  −R_max, +R_max )
```

`sg`는 stop-gradient operator다. `π_teacher`는 domain `d`, reasoning effort `e`의 frozen specialist이고, `π_θ`는 student다. `R_max`는 token별 reward를 상한·하한으로 묶는다.

**수치 예시.** 어떤 token에서 teacher가 확률 0.40, student가 0.05를 준다고 하자. Clip 전 reward는 `log(0.40 / 0.05) = log 8 = 2.08`이다. `R_max = 1.0`이면 reward는 1.0으로 clip된다. Teacher와 student가 일치하는 token(0.30 대 0.30)에서는 `log 1 = 0`이다. Student가 과도하게 생산한 token(teacher 0.02, student 0.50)에서는 `log(0.04) = −3.22`이고 −1.0으로 clip된다. 따라서 signal은 dense하다. 모든 token이 하나씩 signal을 받으며 bounded하다. 이것이 [[kimi-k3]]가 별도 trainer 없이 partial-rollout RL infrastructure를 distillation에도 재사용하는 이유다(§4.1.3).

**group size가 바뀌는 이유.** [[glm-5]]는 distillation 동안 “GRPO algorithm의 group size를 1로 설정해 data throughput을 늘리고 batch size를 1024로 설정한다”고 한다. “advantage가 teacher model과의 gap에서 직접 계산되기 때문”이다(§3.5). [[grpo]]에서 group은 baseline을 추정하기 위해서만 존재한다. Teacher가 baseline을 공급하면 group은 sample 하나로 축소되고 이전 group factor만큼 throughput이 오른다. §3.2의 Reasoning RL 설정인 group 32에서 같은 total sample count로 group 1로 가면 step당 서로 다른 prompt 수가 32배가 된다.

**조건과 한계.** Distillation step만 분리하는 before/after benchmark pair를 출력한 report는 없다. [[deepseek-v3.1]]는 distilled model이 “specialist보다 약간 낮은” 점수를 받고 subsequent RL이 gap을 없앤다고 말하지만 수치는 없다(§3). [[kimi-k3]]는 해당 locus에서 negative result 하나를 보고한다. 더 세밀한 top-k distillation objective는 “convergence speed 또는 final performance에서 명확한 이점”이 없었다(§4.1.3). **해석:** open report 중 어느 것도 equal compute에서 specialist distillation과 mixed stage 하나를 둘 다 실행하지 않았으므로, 현재 둘 사이 선택은 측정된 capability보다 throughput 및 engineering constraint에 근거한다.

## §5 Agentic RL ledger: environment, budget, reward, staleness, context

이 절은 ch-45b가 기계적으로 다룬 setting이 공개 recipe에서 실제로 어떻게 구성되는지 읽는다.

### 5.1 Environment와 prompt

| Report | Environment | Prompt / task | Locus |
|---|---|---|---|
| [[deepseek-v3.1]] | 합성 general-agent environment 1,827개(V3.2에서 pass@100이 0이 아닌 경우 유지) | code-agent 24,667, search-agent 50,275, general-agent 4,417, code-interpreter 5,908 | §3.2.3, Table 1 |
| [[glm-5]] | 9개 언어의 검증 가능 SWE environment 1만 개 이상; Docker build accuracy 90% 초과 terminal environment 수천 개 | 미보고 | §4.2.1--§4.2.2 |
| [[qwen3-coder]] | 독립 environment 20,000개를 병렬 실행 | 미보고 | blog, Post-Training |
| [[qwen3-coder-next]] | environment-building agent가 만든 Docker environment | 9개 이상 언어의 검증 가능 SWE task instance 약 800K | §2.1 |
| [[nemotron-3-super]] | environment 21개, RL dataset 37개; batch당 최대 environment instance 4,000개 | step당 prompt 256개 × response 16개 | §3.2.1, §3.2.4, Figure 12 |
| [[kimi-k2]] | 동시 instance 10,000개 이상 Kubernetes sandbox | 미보고 | §3.2.1 |
| [[kimi-k2-5]] | 동시 agent task 최대 100,000개 Rollout Manager | 미보고 | App. D |
| [[minimax-forge-agent-rl]] | 서로 다른 실제 scaffold와 environment 100,000개 이상 | 하루 sample 수백만 개 | Forge post, opening summary(§1 전) |
| [[deepswe]] | RL iteration당 Docker container 512개 | R2E-Gym problem 4.5K개 | blog §2.1--§2.2 |

**수치 예시: prompt count가 뜻하는 mixture.** [[deepseek-v3.1]] Table 1의 합은 `24,667 + 50,275 + 4,417 + 5,908 = 85,267` prompt이며, §1의 “over 1,800 environments and 85,000 prompts” 진술과 맞는다. 비율은 search agent `50,275 / 85,267 = 59.0%`, code agent `24,667 / 85,267 = 28.9%`, code interpreter `5,908 / 85,267 = 6.9%`, general agent `4,417 / 85,267 = 5.2%`다. 가장 작은 slice는 전부 synthesis로 만든 slice이며, report가 transfer를 보이는 데 쓰는 slice이기도 하다(§4.3, Figure 5). 합성 general-agent task만으로 V3.2-SFT에 RL을 하면 Tau2Bench, MCP-Mark, MCP-Universe가 개선되지만, search와 code environment에서만 RL을 훈련한 V3.2-Exp는 이들에서 개선되지 않는다. Prompt budget 5%가 out-of-domain generalization result를 담당한다.

**수치 예시: batch 산술.** [[nemotron-3-super]] §3.2.4는 step당 prompt 256개, prompt당 response 16개, training batch size 4096을 명시한다. `256 × 16 = 4096`이므로 rollout 하나마다 정확히 gradient update 하나가 생긴다. 따라서 같은 절이 “latest model version보다 최대 한 step 뒤”로 제한한 asynchronous lag까지는 run이 on-policy다. Batch를 바꾸지 않고 group size를 8로 바꾸려면 step당 prompt를 반으로 줄이거나 rollout당 update 두 번을 받아들여야 하며, 후자는 policy lag를 두 배로 만든다.

### 5.2 Algorithm과 off-policy control

모든 2026 recipe는 inference engine distribution과 training engine distribution의 gap을 주 안정성 문제로 다루고, 모두 KL penalty가 아니라 token 또는 sequence masking으로 처리한다.

| Report | Baseline | Off-policy control | KL |
|---|---|---|---|
| [[deepseek-v3.1]] | `Â = R_i − mean(R)`, std division 없음(§3.1) | Off-Policy Sequence Masking: `Â < 0`이고 mean-token `log(π_old/π_θ) > δ`이면 `M = 0`(Eq. 9); Keep Routing; Keep Sampling Mask | unbiased gradient를 위해 K3 estimate에 `π_θ/π_old`를 곱함; strength는 domain마다 다르고 mathematics에는 약하거나 0(§3.1) |
| [[glm-5]] | GRPO group-normalized advantage, group 32, batch 32(§3.2) | `β = 2`, `ε_low = 0.2`, `ε_high = 0.28`인 IcePop `pop(ρ, 1/β, β)`; agentic RL에서 rollout log-probability에 대한 double-sided token clipping + staleness filter | KL term 제거(§3.2) |
| [[kimi-k2-5]] | K1.5-style squared objective(Eq. 1) | advantage 부호와 무관하게 적용하는 token-level log-ratio clip `[α, β]` | objective 안의 `τ log(π_θ/π_old)`(§4.4.2) |
| [[kimi-k3]] | K2.5를 따름(§4.1.2) | partial rollout이 만드는 extreme off-policy regime을 견디도록 한 per-token regularization | K2.5와 같음 |
| [[nemotron-3-super]] | asynchronous GRPO(§3.2.4) | training/inference log-probability의 importance-sampling ratio를 mask; inference worker는 최대 한 step stale | 미보고 |
| [[tongyi-deepresearch]] | leave-one-out advantage(Eq. 4--5)의 GRPO variant | 엄격한 on-policy, ratio를 1.0으로 고정 | 미보고 |
| [[deepswe]] | GRPO++(blog §2.3) | context, step, timeout limit으로 끝난 trajectory를 compact filtering이 mask | KL loss 없음 |

**mechanism: staleness filter.** [[glm-5]] §4.1.2는 response마다 그것을 생성한 rollout model version sequence `(w_0, …, w_k)`를 기록하고, current version `w′`에 대해 `w′ − w_0 > τ`이면 sample을 버린다. 여러 weight push에 걸쳐 생성된 long-horizon trajectory는 큰 importance ratio와 함께 training에 들어오는 대신 training에서 완전히 빠진다.

**수치 예시: filter와 group padding이 batch에 미치는 효과.** `G = 32`, `τ = 4`라 하자. 한 group에서 trajectory 6개가 4 version보다 stale하고 sandbox crash로 5개가 더 실패했다고 하자. Valid sample은 `32 − 6 − 5 = 21`이다. GLM-5 규칙은 valid count가 group 절반을 넘으면 valid sample을 반복해 group을 padding한다. `21 > 16`이므로 중복으로 32까지 채우고 group을 유지한다. 11개 대신 17개가 빠져 15개가 남았다면 group 전체를 버린다. 이 규칙은 selection-biased subset에서 계산한 group-mean baseline이 bias되기 때문에 있다. Crash sandbox는 더 어려운 task와 상관관계가 있으므로 padding 없이 버리면 baseline이 위로 이동하고, 정답이지만 어려운 trajectory를 negative advantage로 바꾼다.

### 5.3 Token budget

세 report는 truncation만 쓰는 대신 reward 안에서 output length를 제어한다.

- [[kimi-k2]]는 task type별 sample당 최대 token budget을 정하고, budget 초과 response는 truncate하고 penalize한다(§3.2.3). Budget과 penalty 값은 출력하지 않는다.
- [[kimi-k2-5]]는 fixed budget의 failure mode를 “length-overfitting”으로 부른다. “rigid budget constraint로 훈련한 model은 더 높은 compute scale로 일반화하지 못하는 경우가 많다.” `m` iteration마다 phase를 번갈아 실행한다(§4.4.2, Eq. 2). Phase 0은 문제 mean accuracy가 `λ`보다 클 때만 budget을 적용하고, Phase 1은 이를 해제한다. Budget 자체는 correct-response length의 `ρ`-th percentile이며 한 번 추정한 뒤 고정한다. K2 Thinking에 보고된 효과는 “Toggle decreases output tokens by 25∼30% with a negligible impact on performance”(Figure 5)다. Training이 mathematics와 programming만 사용했는데도 감소 효과는 GPQA와 MMLU-Pro로 전이한다. **결과(단일 연구)**이며 benchmark별 표는 출력하지 않는다.
- [[kimi-k3]]는 trajectory가 cold-start model에서 문제별로 추정한 budget `b_0(x)`의 `τ · b_0(x)`를 넘으면 task reward를 −1로 override하고, curriculum에 따라 `τ`를 아래로 anneal해 low-, high-, max-effort expert를 만든다(§4.1.2). Agentic task에서는 thinking token만이 아니라 tool-call argument도 count에 포함한다.

**수치 예시.** `b_0(x) = 8,000` token, `τ = 1.5`이면 threshold는 12,000이다. Rollout 8개의 group에서 budget 안 정답 3개(reward 1), budget 안 오답 3개(reward 0), 13,500 token의 정답 2개(reward −1)가 있다고 하자. Group mean은 `(3·1 + 3·0 + 2·(−1)) / 8 = 0.125`다. Budget을 넘었지만 정답인 rollout의 advantage는 `−1 − 0.125 = −1.125`이고, 짧지만 오답인 rollout은 `0 − 0.125 = −0.125`다. 따라서 budget reward는 틀린 것보다 장황함을 더 강하게 penalize한다. Annealing curriculum은 이 행동을 step 0부터가 아니라 점차 도입하도록 설계됐다.

### 5.4 Context policy

Mechanics는 ch-45c가 다룬다. 여기서는 recipe가 선택한 것과 측정한 것을 적는다.

| Report | Policy | 측정 효과 |
|---|---|---|
| [[deepseek-v3.1]] | window 80% 초과 시 trigger되는 Discard-all | 없음 51.4 → 있음 67.6 BrowseComp(§4.4, Figure 6) |
| [[glm-5]] | `k = 5`인 Keep-recent-k, 이후 `T = 32k` 초과시 Discard-all을 쓰는 hierarchical policy | keep-recent만으로 55.3% → 62.0%; hybrid 최종 75.9(§4.2.4) |
| [[kimi-k2-5]] | Agent Swarm: sub-agent가 bounded local context를 가지고 task-relevant output만 반환 | BrowseComp: swarm 78.4 대 single agent 60.6; WideSearch Item-F1 79.0 대 72.7(Table 6) |
| [[minimax-forge-agent-rl]] | context management를 RL loop 안 agent action으로 model화 | 수치 없음; inference-only context management가 “RL training data에서 심각한 distribution shift”를 만든다는 것이 제시 이유 |
| [[tongyi-deepresearch]] | ReAct mode와 함께 SFT에서 context-management mode 훈련 | 분리 ablation 없음 |

행 전체의 pattern은 inference에서만 context policy를 적용해도 BrowseComp에서 6.7에서 16.2점의 가치가 있다는 것이다. 독립된 두 lab([[minimax-forge-agent-rl]], [[tongyi-deepresearch]])은 같은 policy 아래 훈련하면 distribution shift가 사라진다고 주장한다. Equal compute에서 trained context management와 inference-only context management를 비교해 측정한 source는 아직 없다. **Open question.**

## §6 Training target으로서의 parallel orchestration

[[kimi-k2-5]] §3은 single sequential agent가 아니라 orchestration decision 자체를 훈련하는 유일한 공개 report다. 선택 하나하나가 ch-45b에서 다룬 credit-assignment 문제에 대한 응답이므로 정확히 적을 가치가 있다.

1. **Decoupling.** Trainable orchestrator와 “fixed intermediate policy checkpoint에서 instantiate한 frozen subagent”를 결합한다. Subagent trajectory는 objective에서 제외한다. 따라서 subagent output은 differentiable decision이 아니라 environment observation이다. 제시한 동기는 “credit assignment ambiguity and training instability”다. “correct final answer가 flawless subagent execution을 보장하지 않으며, failure도 universal subagent error를 뜻하지 않기” 때문이다.
2. **Reward.** `r_PARL(x, y) = λ1 · r_parallel + λ2 · r_finish + r_perf(x, y)`다. `r_perf`는 task outcome이다. `r_parallel`은 subagent instantiation을 보상해 agent 하나만 계속 실행하는 local optimum인 “serial collapse”에 대응한다. `r_finish`는 완료된 subtask에 보상한다. 이는 meaningful task decomposition 없이 subagent를 많이 spawn하는 것으로 정의한 “spurious parallelism”에 대응한다. `λ1`과 `λ2`는 “훈련 과정에서 0으로 anneal”되어 최종 policy는 task outcome만 최적화한다.
3. **Cost metric.** `CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t))`이며 computation graph의 critical path와 같다. Training과 evaluation budget은 critical step으로 표현한다. 가장 긴 branch를 줄이지 않는 subagent 추가는 아무 것도 얻지 못한다.

**수치 예시.** Main agent가 stage마다 1 step을 쓰고 stage 2에서 6, 9, 4, 7 step을 쓰는 subagent 4개를 spawn하는 3-stage episode를 생각하자. 실행한 총 step은 `1 + (1 + 26) + 1 = 29`다. Critical step은 `1 + (1 + 9) + 1 = 12`다. 같은 subagent 26 step을 subagent 4개에 균등하게 `(7, 7, 6, 6)`으로 나누면 총 work는 같아도 critical step은 `1 + (1 + 7) + 1 = 10`이다. 이 metric은 단순 동시성보다 균형을 보상한다.

**결과.** Table 6에서 swarm은 BrowseComp 78.4%, single-agent는 60.6%다. WideSearch Item-F1은 79.0% 대 72.7%, in-house Swarm Bench는 58.3% 대 41.6%다. Target WideSearch Item-F1에 도달하는 execution time은 “3× ∼ 4.5×” 줄었다. **결과(단일 연구)**다. Lab 하나, model 하나의 결과이며 auxiliary reward 두 항을 분리하는 ablation은 없다.

## §7 Agentic training 뒤 general ability를 재는 report는 무엇인가

Course 목표상 이것이 결정적인 column이지만, 가장 성긴 column이기도 하다.

| Report | Agentic training 뒤 general(non-agentic) evaluation | 근거 |
|---|---|---|
| [[kimi-k2-5]] | 예, text benchmark의 전/후 비교 | visual RL 뒤 MMLU-Pro 84.7 → 86.4, GPQA-Diamond 84.3 → 86.4, LongBench v2 56.7 → 58.9(Table 2) |
| [[qwen3-coder-next]] | 예, base model 대비 | “Qwen-Next와 경쟁력 있으며 MMLU-Redux와 GPQA에서는 약간 개선, MMLU/MMLU-Pro/SuperGPQA에서는 비슷하게 유지”; HMMT25 Feb와 AIME25에서 큰 math gain(§5.3, Table 8--9) |
| [[nemotron-3-super]] | 예, 명시된 training finding | “모든 environment를 동시에 훈련하면 stable gain, single-environment training은 다른 benchmark에서 severe regression” (§3.2.1); §3.3에 전체 benchmark suite |
| [[deepseek-v3.1]] | 부분적: unseen environment만 | MCP-Universe, MCP-Mark, Tool-Decathlon은 RL에 없어서 out-of-domain generalization으로 읽음(§4.1); synthetic-only general-agent RL이 셋 모두로 전이(§4.3, Figure 5) |
| [[glm-4-5]] | qualitative만 | web search와 SWE의 RL은 general tool usage와 Terminal-Bench 같은 coding task를 포함한 “other task와 benchmark 전반의 generalized performance improvement”로 이어진다고 함(§3.3.2); 수치 없음 |
| [[glm-5]] | 간접적: distillation은 측정이 아니라 mechanism | cross-stage distillation의 동기는 “cumulative degradation”(§3.5); 전/후 pair 없음 |
| [[minimax-forge-agent-rl]] | 진술만, 측정 없음 | unified mixed-domain training이 “diverse task 전반 model generalizability를 significantly enhances”한다고 함(§4.1) |
| [[kimi-k2]] | post-agentic 비교 없음 | Table 3은 absolute score; §5는 tool use가 불필요하게 활성화되면 일부 task 성능이 “may decline”한다고 적음 |
| [[deepswe]] | 없음 | post는 SWE-Bench-Verified와 SWE-Bench-Hard validation curve만 평가(blog §4, Figure 2) |
| [[agent-data-protocol]] | 예, central experiment | mixed corpus가 single-domain tuning의 negative transfer를 피함(§6.2, Table 6) |

**반복된 finding.** 독립된 두 source가 task source 다수를 섞은 mixture가 source 자체 task에서 single-source training보다 낫다고 측정한다. SFT의 [[agent-data-protocol]]은 Qwen3-8B에서 SWE-Bench 16.6% mixed 대 SWE-smith-only 11.0%, CodeActInstruct+Code-Feedback-only 0.2%였다. RL의 [[nemotron-3-super]]도 이를 진술하지만 표로 수치를 내지는 않는다. 서로 다른 두 stage와 두 lab에서 일치한다는 점이 이 장의 가장 강한 generality 결과다.

**Harness specificity에 대한 반복 finding.** [[qwen3-coder-next]]는 “한 scaffold trajectory로 훈련한 model은 다른 scaffold로 강하게 전이하지 않는다”고 측정한다. OpenHands는 SWE-Agent로 전이가 약하고, 역방향은 “moderately successful”하다(§3.1.2, Figure 3). [[minimax-m2-aligning-to-what]]는 이를 design post-mortem으로 보고한다. Tool scaling 아래 benchmark score는 올랐으나 “다른 scaffolding framework로 바꾸는 것처럼 environment를 아주 조금만 바꾸면 성능이 급락했다.” 세 recipe가 이에 대응한다. [[kimi-k3]]는 RL 동안 harness configuration을 무작위화하고(§4.2.1), [[nemotron-3-super]]는 OpenHands 안에서 OpenCode와 Codex tool format을 구현해 “multi-harness training이 inference time에 모든 target harness 전반 model generalization과 performance를 개선한다”고 말한다(§3.2.2). [[minimax-forge-agent-rl]]는 gateway를 거쳐 black-box scaffold를 상대로 훈련한다(§2.3).

## 부정 sample과 부정 feedback

Agentic rollout 대부분은 실패한다. Rate를 출력한 유일한 report인 [[swe-smith]] §4는 expert-trajectory 시도 17,906회에서 resolve rate 36%를 기록한다. 따라서 시도의 약 3분의 2는 failure를 낸다. Recipe가 이 failure를 어떻게 다루는지는 측정 가능한 결과를 갖는 recipe 결정이다. ch-43a의 “negative” 네 의미를 쓰면 다음과 같다.

**1. Negative marginal value(discard).** 기본값이다. [[kimi-k2]]는 LLM judge가 task rubric을 통과했다고 판정한 trajectory만 남기고(§3.1.1), [[glm-4-5]]는 multiple judge agent가 complete로 표시한 trajectory만 남긴다(§3.1). [[qwen3-coder-next]]는 termination signal 누락, task failure, malformed tool call을 rule-filter하며(§3.1.2), [[swe-smith]]는 반복적으로 해결된 instance가 “model performance를 저하”시키므로 instance당 trajectory를 3개로 제한한다(§4). Rejection rate는 대체로 미보고이며, 예외는 17,906 attempt에 resolve rate 36%인 [[swe-smith]]다(§4).

**2. Content로서의 negative(cross-entropy로 훈련하되 mask).** [[glm-5]] §3.1은 “trajectory 안 erroneous segment는 유지하지만 loss function에서 mask하여 model이 incorrect action을 강화하지 않고 error-correction behavior를 배울 수 있게 한다”고 쓴다. Failure는 context에 남아 model이 recovery를 보지만, 실패 token은 gradient를 받지 않는다. 이 report들에서 sense 2는 이것이 유일하다.

**3. Conditioning으로서의 negative.** 이 장의 recipe 중 failure를 control token 아래에서 훈련하는 것은 없다.

**4. Gradient로서의 negative.** 네 개의 별도 control이 나타나며 모두 push-down을 제거하기보다 bound한다.

- **Divergence에 의한 sequence masking.** [[deepseek-v3.1]] Eq. 9는 `Â < 0` **그리고** mean token `log(π_old/π_θ)`가 `δ`를 넘을 때 sequence를 zero로 한다. Negative-advantage sequence만 대상이며, 저자들은 “highly off-policy negative sample은 해로울 수 있다”고 한다(§3.1).
- **종료 원인에 따른 trajectory masking.** [[deepswe]] compact filtering은 context limit, step limit, 20분 generation timeout으로 끝난 trajectory를 mask한다. 따라서 중간에 잘린 trajectory는 reward도 punishment도 받지 않는다(blog §2.3).
- **Over-length negative의 선택적 제외.** [[tongyi-deepresearch]] §3.4.3은 이것이 막는 failure를 보고한다. “필터링하지 않은 negative rollout set을 직접 최적화하면 training stability를 유의하게 저하시키며 extended training 뒤 policy collapse로 이어질 수 있다.”
- **Environment-failure 제외.** [[glm-5]] §4.1.2는 sample별 failure reason을 기록하고 sandbox crash를 제거한다. “그런 failure는 model capability가 아니라 environment instability를 반영하므로 noisy training signal을 도입하기” 때문이다.

**Mechanism.** Softmax logit gradient는 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. Token `y`의 negative advantage는 `y`에서 mass를 빼고 alternative에 `p_j` 비례로 재분배한다. Failure가 policy가 아니라 environment 때문에 생겼다면 재분배는 해당 위치에서 model이 원래 말했을 것에 향하며 noise다. 어려운 문제의 timeout이면 push-down은 정답이지만 느린 behavior에 닿는다. 따라서 위 네 control은 reweight하지 않고 mask한다. 부호를 신뢰할 수 없는 sample에서 gradient를 0으로 만들기 때문이다.

**Generality에 미치는 효과.** 이 report들 중 agentic RL 뒤 큰 `k`에서 pass@k를 기록한 것은 없어, coverage loss를 위한 ch-43a 진단은 실행되지 않았다. Pass@k 자체를 보고한 유일한 것은 [[deepswe]]다. RL 뒤 SWE-Bench-Verified에서 Pass@1 42.2%, Pass@16 71.0%지만 비교할 base-model pass@16은 없다(blog §4). **Open question:** [[rlvr-beyond-base-model]]이 single-turn RLVR에서 측정한 방식으로 agentic RL이 solution distribution을 좁히는지는 공개 agentic report에서 아직 시험되지 않았다.

## 설정 recipe

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| GLM-4.5 | 355B / active 32B | mid-train | Agent + long-context stage | 128K에서 100B token(long document up-sample, synthetic agent trajectory 추가) | arXiv:2508.06471v1 §2.3, Figure 3 | 2026-09-15 검증 | ablation 미보고 |
| GLM-5 | 744B / active 40B | mid-train | Context stage와 token | 32K(1T), 128K(500B), 200K(50B) | arXiv:2602.15763v2 §2.3 | 2026-09-15 검증 | 200K stage가 “128K context window 안에서도” 결과 개선; 수치 없음 |
| GLM-5 | 744B / active 40B | mid-train | Issue–PR corpus | issue–PR pair 약 10M; filtering 후 unique token 약 160B | §2.3 | 2026-09-15 검증 | ablation 미보고 |
| GLM-5 | 744B / active 40B | SFT | 최대 context | 202,752 token | §3.1 | 2026-09-15 검증 | ablation 미보고 |
| GLM-5 | 744B / active 40B | RL(reasoning) | Algorithm과 setting | GRPO + IcePop, KL term 없음; β = 2, ε_low = 0.2, ε_high = 0.28; on-policy; group 32; batch 32 | §3.2 | 2026-09-15 검증 | ablation 미보고 |
| GLM-5 | 744B / active 40B | RL(agentic) | Off-policy control | rollout log-probability의 double-sided token clip; staleness filter `w′ − w_0 > τ`; `G/2` 초과 시 group padding을 쓰는 environment-failure exclusion | §4.1.2 | 2026-09-15 검증(τ, ε_ℓ, ε_h 미출력) | stability 개선을 진술; 수치 없음 |
| GLM-5 | 744B / active 40B | RL(agentic) | Environment | 검증 가능 SWE environment 1만 개 이상, 언어 9개; Docker build accuracy 90% 초과 terminal environment 수천 개 | §4.2.1--§4.2.2 | 2026-09-15 검증 | ablation 미보고 |
| GLM-5 | 744B / active 40B | distill | On-policy cross-stage distillation | advantage를 `sg[log(π_teacher/π_θ)]`로 교체; group size 1; batch 1024 | §3.5, Eq. 2 | 2026-09-15 검증 | 전/후 수치 미보고 |
| GLM-5 | 744B / active 40B | eval-gate | Search context management | keep-recent `k = 5`; `T = 32k` 초과 시 Discard-all hybrid | §4.2.4 | 2026-09-15 검증 | keep-recent BrowseComp 55.3% → 62.0%; hybrid 최종 75.9 |
| Kimi K2.5 | 1.04T / active 32B K2 위 구축 | pretrain(joint) | Vision-text joint token | 4K에서 추가 vision-text token 약 15T | arXiv:2602.02276v2 §4.3 | 2026-09-15 검증 | Table 1: 고정 budget에서 early fusion 10%:90%가 mid/late fusion보다 좋음 |
| Kimi K2.5 | 위와 같음 | RL | Token-level clip | advantage 부호와 무관하게 log-ratio가 `[α, β]` 밖인 token의 gradient를 zero | §4.4.2, Eq. 1 | 2026-09-15 검증(α, β 미출력) | long-horizon tool use training stability에 “essential”이라고 진술 |
| Kimi K2.5 | 위와 같음 | RL | Toggle budget schedule | `m` iteration마다 Phase0/Phase1 교대; budget = correct-response length의 `ρ`-th percentile, 추정 뒤 고정 | §4.4.2, Eq. 2 | 2026-09-15 검증(λ, m, ρ 미출력) | Figure 5: K2 Thinking에서 negligible performance impact로 output token 25--30% 감소 |
| Kimi K2.5 | 위와 같음 | RL infra | 동시성 | 동시 agent task 최대 100,000; log-probability를 기록하는 token-in-token-out | App. D | 2026-09-15 검증 | ablation 미보고 |
| Kimi K3 | 2.8T / active 104B | RL | Partial rollout | `N × K` trajectory 중 fraction `λ`가 완료되면 generation pause; pause rollout은 다음 iteration에서 먼저 resume | arXiv:2607.24653v2 §4.1.2 | 2026-09-15 검증(λ, N, K 미출력) | ablation 미보고 |
| Kimi K3 | 2.8T / active 104B | RL | Budget reward | token이 `τ · b_0(x)`를 넘으면 task reward를 −1로 override; domain별 τ anneal | §4.1.2 | 2026-09-15 검증(τ, b_0 estimator 미출력) | low/high/max effort expert 생성; ablation 없음 |
| Kimi K3 | 2.8T / active 104B | distill | MOPD | expert 9개에 대해 token별 `clip(sg[log(π_teacher/π_θ)], −R_max, R_max)` reward | §4.1.3, Eq. 15 | 2026-09-15 검증(R_max 미출력) | top-k objective는 “no clear advantage” |
| DeepSeek-V3.2 | 미출력 | RL | Agent task | code-agent 24,667; search-agent 50,275; general-agent 4,417; code-interpreter 5,908 | arXiv:2512.02556v1 Table 1 | 2026-09-14 검증(card) | §4.3, Figure 5: synthetic-only general-agent RL이 Tau2Bench, MCP-Mark, MCP-Universe 개선 |
| DeepSeek-V3.2 | 미출력 | RL | General-agent environment | 1,827(V3.2에서 pass@100이 0이 아닐 때 유지) | §3.2.3 | 2026-09-14 검증(card) | Table 5: sample task 50개에서 V3.2-Exp pass@1 12% |
| DeepSeek-V3.2 | 미출력 | RL | Off-Policy Sequence Masking | `Â < 0` 및 mean-token `log(π_old/π_θ) > δ`일 때 mask | §3.1, Eq. 9 | 2026-09-14 검증(card); δ 미출력 | 그렇지 않으면 불안정한 run의 stability 개선을 진술 |
| Nemotron 3 Super | 120B / active 12B | SFT | Blend | sample 7M 이상, token 80B; agentic share는 Nano보다 “much larger proportion” | report 2026-04-03 §3.1.2, Figure 12 | 2026-09-15 검증(percentage는 figure에만) | ablation 미보고 |
| Nemotron 3 Super | 120B / active 12B | SFT | Conversational tool use | domain 838개에서 conversation 279,116개(Nano: domain 5개에서 15,588) | §3.1.1 | 2026-09-15 검증 | ablation 미보고 |
| Nemotron 3 Super | 120B / active 12B | RL | RLVR batch | step당 prompt 256 × response 16; batch 4096; rollout당 gradient update 하나; 최대 generation 49K 뒤 64K | §3.2.4 | 2026-09-15 검증 | ablation 미보고 |
| Nemotron 3 Super | 120B / active 12B | RL | Environment | environment 21개, dataset 37개; batch당 environment instance 최대 4,000 | §3.2.1, Figure 12 | 2026-09-15 검증 | all-environment training은 stable gain, single-environment는 다른 곳 severe regression(진술, 표 없음) |
| Nemotron 3 Super | 120B / active 12B | RL | Policy lag | inference worker는 최대 한 step 뒤; weight update 뒤 KV cache 재계산 안 함 | §3.2.4 | 2026-09-15 검증 | ablation 미보고 |
| Qwen3-Coder-Next | 80B / active 3B | mid-train | Repository-level token과 context | token 약 600B; context 32,768 → 262,144; best-fit packing | report 2026-03-03 §3.1.1, §3.2 | 2026-09-15 검증 | Figure 3: mid-training token에 따라 within-scaffold 성능 상승; cross-scaffold transfer 제한 |
| Qwen3-Coder-Next | 80B / active 3B | mid-train | Web-document reformatting | Qwen3-Coder-480B-A35B-Instruct로 rewrite | §3.1.1, Table 1 | 2026-09-15 검증 | Evalplus 54.38 → 63.09; MultiplE 36.02 → 48.35; CRUX-Eval 57.13 → 58.94 |
| Qwen3-Coder-Next | 80B / active 3B | RL | Task pool | 검증 가능 SWE task instance 약 800K, 언어 9개 이상; SFT/RL prompt 완전 분리; pass-rate filtering | §2.1, §4.2.4 | 2026-09-15 검증 | ablation 미보고 |
| Qwen3-Coder-Next | 80B / active 3B | RL | Reward shaping | trajectory-level outcome reward + unfinished-trajectory penalty + invalid tool call의 turn-level token penalty | §4.2.4 | 2026-09-15 검증(값 미출력) | ablation 미보고 |
| Qwen3-Coder | 480B / active 35B | pretrain | Token과 code 비율 | code 70%의 token 7.5T; native context 256K, YaRN으로 1M | blog 2025-07-22 | 2026-09-15 검증 | ablation 미보고 |
| Qwen3-Coder | 480B / active 35B | RL | Environment parallelism | 독립 environment 20,000개 병렬 | blog 2025-07-22 | 2026-09-15 검증 | ablation 미보고 |
| Tongyi DeepResearch | 30.5B / active 3.3B | mid-train | Agentic CPT | 32K 뒤 128K, stage 2에서 64K--128K agentic behavior data; general pre-training data interleave | arXiv:2510.24701 §3.3.1 | 2026-09-15 검증(token 수 미출력) | ablation 미보고 |
| Tongyi DeepResearch | 30.5B / active 3.3B | SFT | Context schedule | stage 1은 40K(all context-management sample); stage 2는 128K + stability용 작은 40K 부분 | §3.4.2 | 2026-09-15 검증 | ablation 미보고 |
| Tongyi DeepResearch | 30.5B / active 3.3B | RL | Algorithm과 reward | 엄격한 on-policy GRPO variant, token-level loss, clip-higher, leave-one-out advantage; binary answer reward, format reward 없음 | §3.4.3, Eq. 4--5 | 2026-09-15 검증(ε 값, G 미출력) | unfiltered negative가 policy collapse, over-length negative 제외 |
| MiniMax-M2.5 | 미출력 | RL | Scale과 algorithm | CISPO; scaffold/environment 100,000개 이상; context 최대 200K; 하루 sample 수백만 | Forge post 2026-02-13 opening summary와 §4.1 | 2026-09-15 검증 | ablation 미보고; post는 benchmark table 미출력(prefix-tree merging은 40× training speedup으로 별도 보고, §3.2) |
| DeepSWE-Preview | 32B | RL | Data와 compute | R2E-Gym problem 4.5K; iteration당 container 512; H100 64개로 6일; step 약 200 | blog 2025-07-02 intro, §2.1--§2.2 | 2026-09-14 검증(card); 공개 script와 충돌 | Figure 2: Pass@1 23% → 42% |
| SWE-smith | dataset | SFT | 비용과 storage | 총 $1360; 사람 노동 약 20시간; repository 128개 instance 50,137개; 295 GB | arXiv:2504.21798 §2.2, Table 2 | 2026-09-15 검증 | Table 3: trajectory 5,016개에서 SWE-agent-LM-32B가 SWE-bench Verified 40.2% |
| Toucan-1.5M | dataset | SFT | Curation | crawl 약 2,800개 중 MCP server 495개 유지; tool 2,000개 이상; 1.5M 중 SFT subset 119.3K | arXiv:2510.01179 §3, §4.1 | 2026-09-15 검증 | Table 2: Qwen2.5-32B-Instruct BFCL V3 61.73% → 70.45% |
| ADP Dataset V1 | dataset | SFT | Composition | dataset 13개, trajectory 1.3M 이상, source 하나가 지배하지 않게 subsample | arXiv:2510.24702v2 §3; mixture weight App. C | 2026-09-15 검증 | Table 6: mixed corpus는 target task에서 single-domain tuning보다 좋고 negative transfer 회피 |

**작은 범용 run의 출발점.** 아래 모든 값은 위 `검증` 행과 source가 쓴 조건에서 왔다. 실행 가능한 environment 몇 개를 가진 7B--32B model이라면, instance당 image 하나가 아니라 repository당 Docker image 하나로 environment를 만든다([[swe-smith]], Python repository 128개, $1360, 295 GB). SFT mixture는 하나가 아니라 여러 task family에서 모은다([[agent-data-protocol]], dataset 13개, trajectory 1.3M, Qwen2.5-Coder 7B--32B). 모든 environment에 걸친 RL stage 하나를 rollout당 gradient update 하나로 실행한다([[nemotron-3-super]], 256 prompt × 16 response = batch 4096, 120B/12B). Context/step/time limit으로 끝난 trajectory는 penalize 대신 mask하고([[deepswe]], Qwen3-32B, problem 4.5K, H100 64개 × 6일), over-length negative는 loss에서 제외한다([[tongyi-deepresearch]], 30.5B / 3.3B). 이 값들을 source가 작은 scale에서 tuning한 것은 아니므로 optimum이 아니라 diagnostic을 기록한 출발점으로 취급한다.

## 일반화 관점

**(a) 무엇이 범위를 넓히는가.**
- **많은 environment에 걸친 mixed RL stage 하나.** [[nemotron-3-super]] §3.2.1은 all-environment training이 stable gain, single-environment training이 다른 곳 severe regression을 낸다고 한다. [[minimax-forge-agent-rl]] §4.1도 같은 이유로 같은 design choice를 보고한다.
- **여러 task family에 걸친 mixed SFT corpus.** [[agent-data-protocol]] §6.2, Table 6에서 Qwen2.5-7B-Instruct의 SWE-Bench는 mixed 10.4% 대 SWE-smith-only 1.0%, GAIA는 9.1% 대 AgentInstruct-only 0.6%다.
- **훈련 중 harness randomization.** [[kimi-k3]] §4.2.1은 고정 harness가 tool schema 하나에 overfit하기 때문에 task group별 harness module을 조합한다. [[nemotron-3-super]] §3.2.2는 multi-harness SWE-RL이 “inference time 모든 target harness 전반 model generalization과 performance를 개선한다”고 보고한다.
- **Agentic mid-training 안 general data interleave.** [[tongyi-deepresearch]] §3.3.1은 foundational generalization 보존을 목적으로 general pre-training data의 작은 비율을 전 과정에 유지한다.
- **가정이 아니라 측정한 cross-domain transfer.** [[kimi-k2-5]] Table 2(visual RL이 MMLU-Pro, GPQA-Diamond, LongBench v2를 올림)와 [[qwen3-coder-next]] §5.3(code reasoning이 competition math로 전이)가 이 장의 측정된 positive transfer 두 개다.

**(b) 무엇이 narrowing 또는 forgetting을 만드는가.**
- **순차적 single-objective stage.** [[glm-5]] §3.5는 distillation stage를 더한 이유로 “cumulative degradation of previously acquired capabilities”를 들며, [[deepseek-v3.1]] §3은 같은 이유로 RL domain을 합친다.
- **Heavy synthetic mid-training data.** [[qwen3-coder-next]] §3.1의 “over-specialization, reduced response diversity, and weaker adaptation to other tasks during fine-tuning.”
- **Tool scaling만 하기.** [[minimax-m2-aligning-to-what]]에서 benchmark score는 오른 뒤 scaffold swap에서 collapse했다. Post는 generalization에 toolset만이 아니라 tool info, system prompt, user prompt, environment, tool response의 perturbation이 필요하다고 결론낸다.
- **Rigid token budget.** [[kimi-k2-5]] §4.4.2의 “length-overfitting”: fixed budget 아래 모델은 “higher compute scale로 일반화하지 못하고” truncated reasoning으로 기운다.
- **필터링하지 않은 negative rollout.** [[tongyi-deepresearch]] §3.4.3: extended training 뒤 policy collapse.
- **불필요한 tool use.** [[kimi-k2]] §5는 필요 없는 곳에서 tool use를 활성화하면 일부 task 성능이 하락한다고 적는다. 이는 model이 tool 사용을 거절할 수 있는 evaluation에서만 보이는 narrowing이다.

**(c) 이 stage에서 재는 법.**
1. **Held-out prompt가 아닌 held-out environment.** [[deepseek-v3.1]] §4.1은 MCP-Universe, MCP-Mark, Tool-Decathlon이 RL에 없었으므로 out-of-domain으로 취급한다. 훈련 environment 안 prompt는 tool schema를 공유하므로 environment 전체를 reserve한다.
2. **Evaluation time의 두 번째 harness.** [[qwen3-coder-next]] §5.1은 모든 model을 SWE-Agent, MiniSWE-Agent, OpenHands에서 300 maximum turn 및 hacking-free protection으로 따로 평가하고 scaffold별 baseline도 복제한다. 같은 model의 scaffold 간 spread가 harness-specificity 측정이다.
3. **Agentic stage 전후 non-agentic benchmark.** [[kimi-k2-5]] Table 2가 template다. 같은 text benchmark를 agentic stage 전 checkpoint와 후 checkpoint에서 실행한다.
4. **Reward-hacking probe.** [[qwen3-coder-next]] §4.2.4는 git remote 제거만으로 부족함을 보인다. Agent는 `git remote add`, `git clone`, `curl`로 재연결했다. Blocked tool call을 일회성 audit이 아닌 rate로 기록한다.
5. **Coverage.** 큰 `k`의 pass@1과 pass@k를 기록한다. 이 장의 agentic report 중 [[deepswe]] pass@16을 넘어 이 작업을 한 것은 없으므로 진단은 가능하지만 사용되지 않았다.

## 흔한 실수와 발견 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| Agent harness 하나에서 훈련·평가 | training scaffold 점수는 높지만 같은 model의 다른 scaffold 점수는 훨씬 낮음 | 동일 turn limit으로 scaffold 최소 두 개에서 평가([[qwen3-coder-next]] §5.1); spread 보고 |
| Trainer 쪽에서 rollout text를 다시 tokenize | Loss/reward는 정상처럼 보이나 tool-call boundary 근처 잘못된 token에 gradient 연결 | token-in-token-out 채택, sample에서 저장 rollout token id와 retokenized text 비교([[glm-5]] §4.1.2, [[kimi-k2-5]] App. D) |
| Sandbox crash를 model failure로 채점 | 특정 repository에서 reward variance spike; reward가 difficulty보다 environment와 상관 | sample별 failure reason 기록, environment collapse 제외([[glm-5]] §4.1.2) |
| Invalid sample을 group padding 없이 버림 | Group baseline 위로 drift; 정답이지만 어려운 trajectory가 negative advantage | valid sample이 `G/2` 넘으면 group padding, 아니면 group 전체 drop([[glm-5]] §4.1.2) |
| Greedy asynchronous scheduling | 초기에는 짧고 쉬운 task가 지배, 이후 hard task가 군집; distributional 이유로 reward curve non-monotonic | generation queue의 bounded visibility window([[minimax-forge-agent-rl]] §3.1) 또는 staleness threshold([[glm-5]] §4.1.2) |
| Inference에서만 context management 적용 | context fold 직후 agent behavior 불일치; training reward가 deployed accuracy 예측 못함 | 같은 context policy 아래 훈련하거나 context management를 action으로 model화([[minimax-forge-agent-rl]] §2.2) |
| Step 0부터 fixed token budget | output length가 하락한 채 유지; model이 더 큰 inference budget을 못 씀 | budget-limited/free phase 교대([[kimi-k2-5]] Eq. 2); 두 inference budget에서 accuracy 측정 |
| SFT prompt를 RL prompt로 재사용 | Benchmark movement 없이 RL reward 즉시 상승 | SFT와 RL prompt set 분리([[qwen3-coder-next]] §4.2.1), pass rate filtering |
| SWE environment에서 future commit이 reachable | repository network access instance의 비정상적으로 높은 resolve rate | repository URL과 git/curl/wget을 결합한 tool call block, block rate 기록([[qwen3-coder-next]] §4.2.4) |
| SFT-effective dataset을 RL prompt pool로 취급 | 높은 solve-none rate; 대부분 group의 advantage 0 | Run 전 current policy 아래 pass-rate distribution 측정([[deepswe]] §6) |

## 이해 점검

1. GLM-5는 GLM-4.5보다 total token budget을 1.24배, agent 포함 mid-training token을 5.5배(100B → 550B), 그 token이 budget에서 차지하는 비율을 약 4.4배(0.43% → 1.9%) 늘린다. 셋 중 agentic capability를 예측할 수 있는 수치가 무엇인지 설명하고, 이를 확인하려면 어떤 측정이 필요한지 말하라.
2. DeepSeek-V3.2는 agent prompt의 5.2%를 합성 general-agent task에 쓰고 바로 그 slice로 out-of-domain generalization을 보인다. 합성 slice가 transfer result에 자연스러운 위치인 이유와, 그 해석을 falsify할 결과를 설명하라.
3. GLM-5와 Kimi K3는 distillation에서 group-relative advantage를 teacher log-ratio로 바꾼다. 이때 group size를 1로 내릴 수 있는 이유를 유도하고, 그렇게 할 때 잃는 것을 말하라.
4. Nemotron 3 Super는 throughput 때문에 multi-environment RLVR에서 SWE-RL을 분리하면서도, single-environment training은 다른 곳 regression을 낸다고 보고한다. 두 진술은 긴장 관계인가? 이 pipeline에서 분리가 가능한 이유를 설명하라.
5. DeepSeek-V3.2 sequence masking, DeepSWE compact filtering, Tongyi over-length exclusion, GLM-5 environment-failure exclusion의 네 report는 negative sample을 down-weight하지 않고 mask한다. Softmax logit gradient를 이용해 signal 부호가 신뢰할 수 없을 때 masking이 맞는 이유와 down-weighting이 동등하지 않은 이유를 설명하라.
6. Qwen3-Coder-Next는 synthetic mid-training data의 상한을 말하고 Kimi K2.5는 fixed budget 아래 length-overfitting을 말한다. 서로 다른 stage의 over-specialization failure인 둘의 공통 mechanism을 한 문단으로 구성하라.
7. Kimi K2.5 PARL은 training 동안 `λ1`, `λ2`를 0으로 anneal한다. 계속 상수로 두면 무엇이 잘못되며, step 0부터 0이면 무엇이 잘못되는지 설명하라.
8. §7의 report 중 agentic stage 전후 non-agentic evaluation을 실행한 것은 세 개뿐이다. Lab이 agentic training이 model을 narrow하지 않았다고 주장하려면 필요한 최소 evaluation protocol을 설계하고, 공개 근거상 이 장의 어느 recipe가 이를 통과할지 말하라.

## 연결

- **이전 장:** ch-45c — Context Management for Long-Horizon Agents. 여기 §5.4는 그 장이 유도한 context policy를 각 lab이 보고한 BrowseComp delta와 함께 recipe selection으로 읽는다.
- **의존성:** ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data. 여기 §4는 그 장 질문의 agentic special case이며 on-policy variant를 추가한다.
- **다음 장:** ch-46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention. 그 lab의 held-out-retention design은 대부분 report가 비워 둔 §7 column의 small-scale version이다.
- Track 앞부분: ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability(§5에서 구성한 mechanics); ch-45a — Preference-Optimization and RL Stage Recipes Side by Side(non-agentic stage의 같은 cross-lab reading); ch-45 — Self-Improvement Loops and Multi-Stage Reasoning Pipelines(GLM-4.5의 iterative self-distillation).
- Earlier stage: ch-32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training(여기 §2); ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares(여기 §3); ch-29c — Agentic Environment and Task Synthesis at Scale 및 ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories(여기 §3 및 §5.1).

## 출처

- [[kimi-k2]], [[kimi-k2-recipe]] — joint RL stage 하나, MCP와 synthetic tool synthesis, budget control, PTX loss, 10,000-instance sandbox; stage map의 2025 baseline.
- [[kimi-k2-agentic-data]] — K2 tool-use synthesis 절을 독립적으로 읽음.
- [[kimi-k2-5]] — zero-vision SFT, token-level log-ratio clipping, Toggle budget schedule, PARL/Agent Swarm, 그리고 이 장의 유일한 전/후 general-benchmark 표.
- [[kimi-k3]] — domain 3개·reasoning effort 3개를 가로지르는 expert 9개, MOPD, iteration 간 partial rollout, budget reward −1, unified white-box harness environment.
- [[glm-4-5]], [[glm-4-5-recipe]] — unified SFT로 distill한 expert 3개, 100B-token agent mid-training stage, search/SWE agentic RL과 format-violation reward 0.
- [[glm-5]] — 200K까지 세 stage mid-training, staleness filtering과 token-in-token-out을 쓰는 asynchronous agentic RL, on-policy cross-stage distillation, keep-recent-k context management.
- [[deepseek-v3.1]], [[deepseek-v3.1-recipe]] — specialist distillation 뒤 mixed RL stage 하나, Table 1 agent-task count, Off-Policy Sequence Masking, unseen-environment 근거.
- [[qwen3-coder]] — code 70%의 token 7.5T, 병렬 environment 20,000개; 이 집합 agentic RL infrastructure의 가장 이른 scale point.
- [[qwen3-coder-next]] — minimum-synthetic-data principle, cross-scaffold transfer 측정, model 하나로 distill한 expert 4개, reward-hacking blocker.
- [[nemotron-3-super]] — 가장 상세히 공개한 post-training blend: SFT sample 7M, RL environment 21개, all-environment 대 single-environment finding, PivotRL.
- [[tongyi-deepresearch]] — 이름 붙인 mid-training stage로서 Agentic CPT, environment 세 유형, selective negative exclusion을 쓴 엄격한 on-policy RL.
- [[minimax-m2-aligning-to-what]] — tool-scaling negative result와 agent generalization의 perturbation 관점.
- [[minimax-forge-agent-rl]] — unified mixed-domain CISPO, action으로서 context management, windowed FIFO scheduling, prefix-tree merging.
- [[deepswe]], [[deepswe-recipe]] — GRPO++와 compact filtering을 쓰는 RL-only training; 이 장에서 agentic RL 뒤 pass@k 수치를 낸 유일한 source.
- [[toucan-mcp]] — MCP trajectory 1.5M과 SFT subset 119.3K의 BFCL V3 gain.
- [[agent-data-protocol]] — interlingua schema 및 single-domain tuning의 negative transfer를 재는 cross-task transfer 표.
- [[swe-smith]] — repository 128개에서 $1360 및 295 GB로 만든 instance 50,137개; 반복 해결 instance에 대한 trajectory-capping 규칙.
- [[r2e-gym]], [[swe-gym]] — [[swe-smith]] storage/cost 비교의 executable-environment baseline.
- [[grpo]] — §4 distillation objective가 대체하는 group-relative baseline.
- [[rlvr-beyond-base-model]] — 이 장의 어떤 agentic report도 답하지 못한 pass@k coverage 질문.
