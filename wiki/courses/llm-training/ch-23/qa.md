<!-- chapter: ch-23 — Model Collapse and Synthetic-Data Verification
     deps: [[read]]
     scope: clarifying questions captured during Read phase. Kernel-only — full explanations stay in read.md / discuss transcript.
-->

# Ch-23 — Q&A

Back to [[read]]. Cross-chapter links: [[../ch-21/qa]], [[../ch-22/qa]].

---

## Q1 — "Never replace real data with synthetic" vs Phi rewriting pretrain data?

**Question**: ch-23 says *never replace real data with synthetic*, but ch-21 covered Phi rewriting pretrain into textbook-style. Contradiction? Does the rule only apply to SFT?

**Kernel**:

핵심 단어는 **replace, not augment**. 가이드라인은 *all training stages* (pretrain + SFT + RL)에 적용 — ch-23 model-collapse paper는 사실 pretraining 무대.

Collapse 발생 3-condition (모두 동시):
1. Real anchor 제거 (grounding 없음)
2. Recursive generation (Gen-n trains on Gen-(n-1) output)
3. Verification gate 없음

**Phi가 위반 안 한 이유**:
- Filtered real web (The Stack code + filtered web ~1.3B tokens) = real anchor 유지
- Synthetic textbook = *augmentation layer*, not replacement
- Single-generation (Phi-1을 Phi-1.5 generator로 사용 안 함) → recursive loop 없음

→ ch-21 [[../ch-21/qa]] Q9 (all-textbook fail) + Q10 (bootstrapping paradox) 답이 정확히 ch-23 thesis. Phi는 "textbook + filtered real" 혼합이지 *pure textbook이 아님*.

**Stage별 collapse 영향**:
- Pretrain: 모델 base distribution 깨짐 (가장 위험)
- SFT: instruction distribution 좁아짐 (mix recoverable)
- RL: reward hacking + mode collapse (verifier mitigation → ch-44 RLVR)

**Strong model collapse 추가 발견 (Dohmatob et al.)**: 1% pure synthetic contamination만으로도 scaling law break → scaling 무한대 가도 not recoverable.

---

## Q2 — `E[R_test] ~ f(N)` 표기법 의미?

**Question**: E, R_test, f, N 각각 뭐?

**Kernel**:

| 기호 | 의미 |
|---|---|
| `E[·]` | Expectation. Training data sampling + test sampling + init randomness 모두 평균 |
| `R_test` | Test risk = test loss/error. `R = Risk` in stat learning theory |
| `N` | Training set size (보통 # tokens). Chinchilla/Kaplan convention |
| `f(N)` | scaling-law function. Canonical: `A · N^(-α) + ε` |
| `~` | "scales as", asymptotic equality up to constants |

**Canonical scaling law**:
$$\mathbb{E}[R_{test}] \sim A \cdot N^{-\alpha} + \epsilon$$

- α = scaling exponent (0.3~0.5)
- ε = **irreducible error floor** (모델 크기 무한대여도 안 줄어듦)

**ch-23 thesis로의 연결**:
- Clean data: `E[R_test]`가 power-law로 감소 (scaling holds)
- Synthetic contamination 1%만 섞여도: power-law term `N^(-α)` 사라지고 ε floor가 훨씬 높아짐 → scaling law *break*
- 즉 `E[R_test] ~ f(N)` 관계 자체가 *오직 clean data 조건에서만* 성립

→ Strong model collapse 정리: scaling N→∞ 가도 recover 불가.

---

## Q3 — `f(N) + c(p) · σ_synth²`에서 c와 p?

**Question**: c와 p가 뭐?

**Kernel**:

| 기호 | 의미 |
|---|---|
| `p` | training mixture 중 **synthetic data 비율** (0 ~ 1) |
| `c(p)` | contamination 계수, p의 함수. 대표 form: `c(p) = p / (1-p)` 또는 `c(p) ∝ p` |
| `σ_synth²` | synthetic distribution이 real과 얼마나 다른지 (bias/variance term). Perfect generator → 0, mode-collapsed generator → 큼 |

**c(p) properties**:
- `c(0) = 0` (synthetic 없으면 contamination term 사라짐)
- `c(p) > 0` for any `p > 0`
- `p → 1`에서 발산

**전체 식**:
$$\mathbb{E}[R_{test}] \sim \underbrace{f(N)}_{\text{scales with data}} + \underbrace{c(p) \cdot \sigma_{synth}^2}_{\text{irreducible floor, N-independent}}$$

**Strong collapse 결론**: 두 번째 term은 `N`과 무관 → `N → ∞`여도 `E[R_test] → c(p) · σ_synth² > 0`. Clean-data 모델 성능 *도달 불가*.

**ch-23 verifier로의 연결**: filter는 단순 quality 거르기 아니라 **σ_synth² → 0 근사** (distribution-preserving)이어야 의미 있음. ch-22 §5 factual correctness 한계가 ch-23 forward link 되는 이유.

---

## Q4 — Rare-token PPL 상승 = rare token이 "사라지는" 것?

**Question**: Tail collapse 결과 rare-token PPL이 높아진다는 것이, 그 token이 model에서 "잊혀지는/사라지는" 것이라고 볼 수 있는가?

**Kernel**:

세 layer로 구분 — 부분적으로 yes, 정확히는 *output distribution에서 push out*.

| Layer | 사라지는가? |
|---|---|
| **Vocab/tokenizer** | ❌ 그대로 존재. Input encoder도 인식 가능 |
| **Probability mass** | 재분배. P(rare\|ctx) ↓ → modes(common tokens)로 이동 |
| **Generation output** | ✅ 사실상 absent. Greedy / top-p / top-k 모두 candidate에서 cut off |

**일반 "catastrophic forgetting"과 차이**:
- Catastrophic forgetting: fine-tuning으로 *representational knowledge* 손실 → rehearsal로 회복 가능
- Tail collapse: **distribution 모양**이 좁아짐. Single-gen은 bias, **recursive across generations**일 때 catastrophic. Real anchor 없으면 매 generation마다 tail 깎임 → eventual 완전 absence.

**Recursive mechanism (왜 catastrophic)**:
Generator가 perfect해도 발생. Finite sampling이 low-probability를 *under-sample* → 각 generation이 *upstream sampling noise*를 truth로 학습 → tail 누적 손실. (approximation error = 0이어도 *statistical error* > 0)

**[[../ch-22/qa]] verdict E2 연결**: GPT-4/Claude가 저자원어 tail이 이미 collapsed → 그들의 generation 자체가 tail-thin → student는 첫 generation부터 collapsed tail 상속. Real corpus anchor가 *grounding*으로 반드시 필요한 이유.

---

## Q5 — "Gate vs No Gate" 의미?

**Question**: Gate vs no Gate 차이?

**Kernel**:

**Gate** = synthetic data가 training pool에 들어가기 전 통과해야 하는 verification filter. ch-23 core thesis가 이 한 단어에 압축됨.

| | No Gate | Gate |
|---|---|---|
| Path | Generator → Pool 직접 | Generator → [Verifier] → Pool |
| σ_synth² | 큼 (generator bias 그대로) | → 0 (faithful sample만 통과) |
| Q3 floor `c(p)·σ_synth²` | 큼 → collapse | 최소화 → scaling 회복 |
| Default outcome | **Model collapse (가만히 두면 무조건)** | Mitigation 가능 |

**Gate 종류**:
- Exact-match verifier (math/code): HIGH strength
- Round-trip consistency: MED-HIGH
- LLM-as-judge: MED (judge bias inheritance 위험 — [[../ch-22/qa]])
- Faithful-synth-eval (statistical alignment): MED
- Reward model: MED, RM-bias 위험
- 없음: NONE (collapse default)

**ch-22 selection methods와 결정적 차이**:
- Selection (ch-22) 목적 = quality/informativeness/diversity *고르기*. σ_synth² 직접 target 아님
- Gate (ch-23) 목적 = distribution-faithful *만 통과*. σ_synth² 직접 감소
- → ch-22 §5 factual correctness 한계가 ch-23 forward link 되는 지점. **Selection ≠ Gate.**

**Pipeline 위치**: Verifier (cheap, gate) → Selector (expensive, ch-22) → Pool. ch-22 Stage 4 cheap-first ordering이 ch-23에서 *normative*화.





---

## Q-정정 (2026-09 revision)

2026-09 revision에서 read.md를 primary source 기준으로 다시 작성했다. 아래 Q의 kernel에 이제 정정된 사실이 들어 있다. 이전 entry의 line reference(예: "read.md L156")는 git commit 4a72e54 시점의 read.md를 가리킨다.

- **Q1.** Shumailov et al.의 LLM 실험은 pretraining이 아니라 OPT-125m을 wikitext2로 fine-tuning한 실험이다. "real anchor 제거 + recursion + verification gate 없음"이 모두 있어야 collapse라는 정리는 정확하지 않다: verifier 없이 real data와 synthetic data를 accumulate만 해도 test error가 bounded로 유지된다. 이 결과는 9M–126M language model(TinyStories)과 linear regression에서 측정되었고, Gemma 2 SFT에서 재현되었다(read.md §2, [[model-collapse-accumulation]] Table 2, [[collapse-or-thrive]] Fig. 3). "1% synthetic만으로 scaling law break"는 label-shift linear regression theory의 결론이고, GPT-2 실험에서는 scaling이 늦어지는 것만 보였고 plateau는 보이지 않았다(§3). phi-1 data는 filtered code-language 약 6B tokens + synthetic textbook 1B tokens 미만이며, 1.3B는 parameter 수이다([[phi-textbooks]] §2 기준).
- **Q2.** "α = 0.3~0.5"는 source가 없다. Synthetic data가 섞여도 `σ²d/n` term은 n에 따라 계속 줄어들고, n과 무관한 floor `p₂²c²`가 더해진다(Corollary 1, Eq. 12). Power-law term이 사라지는 것이 아니다.
- **Q3.** `c(p) = p/(1−p)`, `σ_synth²`, "p → 1에서 발산"은 논문의 식이 아니다. 논문의 floor는 `p₂²c²`이고, `p₂`는 synthetic fraction, `c²`는 synthetic label function과 real label function의 mismatch(quality)이다.
- **Q4.** 논문은 rare-token perplexity를 측정하지 않았다. Real test set의 mean perplexity는 generation마다 올라가고, generation-0 model로 채점한 generated sequence 분포가 high-probability 쪽으로 이동하면서 긴 tail이 생긴다(Fig. 1). "GPT-4/Claude의 저자원어 tail이 이미 collapsed"라는 주장은 확인된 source가 없다(Open question).
- **Q5.** "No gate면 무조건 collapse"는 틀렸다: accumulate는 verifier 없이 collapse를 막는다. Verifier는 `σ_synth²`를 0으로 만드는 장치가 아니라, 정답 keep rate ϕ와 오답 keep rate ψ의 비율로 breakdown point `p* = 1/(1 + ψ/ϕ)`를 올리는 장치이고, 여러 round 반복하면 model은 verifier의 knowledge center로 수렴한다(§4). Gate 강도 목록(faithful-synth-eval 포함)은 source가 없고, exact-match verifier도 false positive(정답이지만 틀린 reasoning)와 false negative(parser가 정답을 버림, [[bespoke-stratos]])를 가진다.
