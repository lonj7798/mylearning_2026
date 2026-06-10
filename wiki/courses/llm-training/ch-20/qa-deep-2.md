<!-- chapter: ch-20 Q&A part 3; deps: [[read]], [[qa]], [[qa-deep]]; kernel answers only -->
# Ch-20 — Reading Q&A (Part 3: weight sharing / control axes / λ balance)

Continued from [[qa-deep]]. Part 1 ([[qa]]) covers rollout vocab; Part 2 ([[qa-deep]]) covers Orca / Orca-2 / DSBS mechanisms.

## Q5. "모델 weight 하나만 있음"의 의미 + DSBS의 hint는 Orca의 hint와 어떻게 다른가?

**Weight sharing**: 별도 두 모델(label용 770M + rationale용 770M = 1540M total) *아님*. **한 set의 770M parameter가 두 task 모두 담당**. Training 시 두 task의 gradient가 *같은 weight에 흘러들어와 합쳐짐*. 비유: 의사 두 명 고용하는 게 아니라 한 의사가 내·외과 둘 다 봄.

**왜 sharing이 핵심**: 만약 별도 모델이었다면 label 모델은 rationale의 학습 압박을 못 받음 → label-only baseline과 똑같이 학습. Sharing 덕분에 rationale task의 gradient가 encoder weight을 *reasoning feature를 encode하도록* 밀어줌 → 그 weight으로 label task도 처리되니까 label 성능도 올라감.

**"[label]/[rationale]도 hint잖아?"** — 기술적으로 hint이지만 Orca의 hint와 *function이 다름*:

| | Orca system message | DSBS prefix tag |
|---|---|---|
| 누구한테 | Teacher | Student |
| 무엇을 통제 | Trace의 *style / form* | 두 *다른 task* 구분 |
| 같은 task vs 다른 task | 같은 task의 surface form 다양화 | 두 분리된 task |

**Inference 시 hint axis**:

| | Inference 시 hint | Student auto-select |
|---|---|---|
| Orca v1 | 없음 | Strategy (실패, 항상 long trace) |
| Orca-2 | 없음 (erased) | Strategy (성공 via prompt erasing) |
| **DSBS** | **항상 `[label]`** | **None — task=label 하나뿐** |
| R1 | 없음 | Strategy + depth (RL이 형성) |

→ DSBS는 *strategy auto-selection* 게임에 참가 안 함. Inference task가 항상 label이라 auto-select 불필요. Prefix가 inference에 남아있어도 cheating 아님.

**DSBS가 Orca보다 "뛰어난가"는 axis-dependent**: Synthetic data 생성은 거의 동일 (teacher CoT). 데이터 *사용*이 다름. DSBS 우위(latency, parameter efficiency, clean control), Orca 우위(instruction 범용성, explainability). 2025 R1 verdict: DSBS thesis + Orca always-emit + verifier = 셋 다 결합.

## Q6. Distillation은 reasoning을 *통제하는* 방법론 — control axis framework

각 method가 *다른 control axis*를 잡음. Method의 성능 = 어떤 axis를 prioritize했는가.

| Method | Control axis | 무엇을 통제 | 통제 못 하는 것 (failure mode) |
|---|---|---|---|
| Orca v1 | Reasoning expression *style* | 16 system msg로 trace surface form | 언제 emit할지 ("2+2?"에도 400 token) |
| DSBS | Training-vs-inference *separation* | Training엔 aux, inference엔 숨김 | Inference 시 explanation 없음 |
| Orca-2 | Strategy *selection* | Prompt erasing으로 student가 자체 결정 | Reasoning *quality* — teacher ceiling 못 넘음 |
| R1 | Reasoning *emergence + grounding* | RL이 verifier 압박 하에 reasoning 발견 | Style tics ("Wait, reconsider"가 trivial 질문에도) |

**진화 패턴**: 새 method가 old method의 *통제 못 했던 axis*를 잡으면서, 자기는 다른 axis를 unrolled로 남김.

```
Orca v1 → DSBS:    style 다양화는 됐는데, inference에 trace 안 emit은? → DSBS 해결
Orca v1 → Orca-2:  style 다양화는 됐는데, 언제 어떤 strategy? → Orca-2 해결
Orca-2 → R1:       strategy selection은 됐는데, ceiling 넘기? → R1이 RL+verifier로 해결
```

**핵심 통찰**: "가장 우월한 method냐"라는 질문 자체가 wrong framing. 정답: **control 우선순위에 따라 다름**. Latency 중요 → DSBS. Explainability → Orca/R1. Scale + ceiling → R1. Hand-crafting 제거 → R1.

## Q7. DSBS ablation — `[label]+[rationale]` training은 정말 도움 되나? λ의 golden balance?

**Q1: [label]만 inference 시 도움 되나** — YES, 일관되게.

| Benchmark | Label-only T5-770M | DSBS T5-770M | Δ |
|---|---|---|---|
| ANLI | ~50% | ~58-60% | +8~10pt |
| e-SNLI | ~88% | ~92% | +4pt |
| CQA | ~62% | ~67-70% | +5~8pt |
| SVAMP | ~50% | ~66% | **+16pt** (= PaLM-540B few-shot) |

SVAMP에서 max gain — math는 intermediate step이 결정적, rationale task가 encoder에 step-decomposition feature를 가장 강하게 encoding.

**Rationale annotation density ablation**: 0% → 100%로 갈수록 *monotonic 증가*, diminishing return 거의 없음. [[ch-19]]의 [[s1]]/[[limo]] *less is more*와는 직교 axis (그건 total sample 수, 이건 sample당 annotation 풍부도).

**λ golden balance — universal 값 없음**. Benchmark별 0.3 ~ 1.0, ~0.5 typical:

| Benchmark | Best λ |
|---|---|
| ANLI | ~0.5 |
| e-SNLI | ~1.0 (rationale=explanation 자체) |
| CQA | ~0.3 |
| SVAMP | ~0.5 |

**λ U-curve**:
- λ → 0: rationale gradient 거의 없음 → label-only baseline과 동일
- λ 적정 (~0.5): encoder regularization 강함, label decode는 유지 → peak
- λ → ∞: rationale에 capacity over-allocated, label accuracy *떨어짐* (label prefix에도 trace 같은 거 뱉음)

**진짜 의미**: λ ≈ `(L_label gradient norm) / (L_rationale gradient norm)`. Rationale은 output sequence 길어서 gradient 자연스럽게 큼 → λ<1로 normalize. Multi-task learning의 일반 원리 (GradNorm 등으로 자동 tune 가능).

**R1에서는 λ tuning이 *사라짐***: trace + answer를 한 autoregressive sequence로 묶음 → loss 하나뿐. Token 수에 비례해 자연스럽게 weight (long trace = 더 많은 gradient). Architecture 단순화 + λ-free. DSBS의 multi-task framework가 deprecate된 또 하나의 측면.

## Q12. OpenThoughts는 single QA pair인가? — 두 axis 분리

**Axis 1 — Sample format**: YES, single-turn (problem, trace, answer) tuple. Multi-turn dialogue/tool-calling 아님. R1-distill 계열 다 동일.

**Axis 2 — Unique problem당 rollout 수**: NO, 같은 problem을 *여러 번* sampling.

```
Stratos:       17K unique × 1   = 17K samples
Open-R1:       220K unique × 2  = 440K samples
OpenThoughts: ~75K unique × 16 ≈ 1.2M samples
```

OpenThoughts는 QwQ teacher에 T>0 sampling으로 같은 problem을 16번 굴림 → 16개 *다른 reasoning path* (같은 답이라도 풀이 다름). Line 179: *"sampling multiple answers per question is the easiest diversity trick — ≥16× expansion per source with non-trivial gain"*.

**왜 이게 최고 diversity trick**: (1) 새 problem curate 비용 0, teacher inference cost만; (2) 같은 problem의 16가지 풀이 = student가 *move repertoire*를 학습 (problem instance memorize가 아니라); (3) **pass-rate selection** — 16 rollout 중 verifier 통과 비율 = `pass@16` = difficulty signal.

**ch-19 pass@k framework의 직접 instantiation**: 너의 verdict E6에서 reinvent한 *"generate same problem multiple times → pass@k as verifier + difficulty signal"*이 정확히 OpenThoughts의 sampling axis. 너는 논리로 도출, OpenThoughts는 1000+ ablation으로 empirically 확인.

**Breadth vs depth at *which axis***: 같은 token budget에서 *sampling axis 확장 > problem axis 확장*. 직관과 약간 반대 — "더 다양한 problem"이 아니라 "한 problem의 다양한 풀이"가 더 강한 학습 signal. Reasoning은 problem memorization이 아니라 move internalize이라서.

**Q9+Q11+Q12 종합 — OpenThoughts recipe = 3-axis 동시 optimization**: (1) Teacher = QwQ (distribution match), (2) trace length = median 3K (token budget 효율), (3) sample-per-problem = 16 (move diversity). 모두 R1-distill의 default와 반대 선택 → AIME25 53%로 strongest open-data 7B reasoning model.

## Q13. Single-turn SFT가 multi-turn 성능을 boost하나? — Partial yes, partial regression

**Transfer 됨**: per-turn reasoning quality. `<think>` behavior는 turn boundary 무관.

**Transfer 안 됨 (4 axis)**: (a) dialogue coherence, (b) context aggregation, (c) tool-calling format, (d) conversational style/refusal — 모두 base instruct model의 prior.

**Regression 패턴** (Qwen2.5-7B-Instruct → R1-distill): AIME25 18→53 (+35), MT-Bench 7.2→5.8 (-1.4), IFEval 72→54 (-18). Reasoning gain ↔ dialogue/IF loss의 직접 trade-off.

**Mechanism**: (1) format prior shift — single-turn prior가 multi-turn에도 emit; (2) capacity allocation — 7B는 reasoning에 capacity 옮기면 dialogue prior 흐려짐; (3) distribution mismatch — R1 trace에 dialogue/tool pattern 없음.

**Mitigations**: replay mixing / two-stage SFT / mixed-format embedding / production routing. **Parked Qs**: ch-19 deferred sales-call ([[ch-25]]) + tool-calling ([[ch-26]])가 정확히 R1-distill blind spot. **Control axis 8** (Q6 확장): multi-turn / format preservation — uncontrolled in R1-distill, 2025 frontier.
