<!-- chapter: ch-30 — SFT Design Axes
     companion to: [[read]]
     kind: qa
     track: sft
-->

# ch-30 — Reading Q&A

## Q1. Qwen 템플릿에 BOS가 없는데, 이게 중요한가?

**Kernel.** BOS의 부재 자체는 무해하다. BOS는 의미를 나르는 토큰이 아니라 position-0에서
attention sink 역할을 하는 자리표시자이고, Qwen은 `<\|im_start\|>`가 그 자리를 대신한다.
중요한 건 "BOS가 있냐"가 아니라 **pretraining이 본 position-0과 SFT/inference의 position-0이
일치하냐**다. 그리고 실제 버그는 비대칭적으로 BOS가 *있는* 쪽에서 난다.

| | Qwen (`bos_token: null`) | Llama-3 (`<\|begin_of_text\|>`, `add_bos_token=true`) |
|---|---|---|
| jinja 템플릿에 BOS | 없음 | `{{ bos_token }}` 포함 |
| `tokenize=False` → 재토크나이즈 | 안전 (넣을 BOS가 없음) | **double-BOS** (`add_special_tokens` 기본 True) |
| packed batch의 문서 경계 | `<\|im_start\|>` — turn 경계와 동일 토큰, 구분 불가 | `<\|begin_of_text\|>` — 육안 식별 가능 |

2행이 read.md L134의 "always call `apply_chat_template(tokenize=True)`" 권고가 막는 바로 그 버그.
Qwen은 이 footgun이 구조적으로 불가능하다 — 없는 게 오히려 안전한 방향.

3행이 ch-30이 말하지 않은 부분: §3의 검증 절차("decode a packed batch and eyeball the
delimiters", L87)가 **Qwen에서는 더 약하다**. packing 버그(block-diagonal mask 누락,
position-ID reset 누락)가 났을 때 Llama-3는 스트림 중간의 `<\|begin_of_text\|>`로 seam이 보이지만,
Qwen은 문서 경계와 turn 경계가 같은 토큰이라 decode해도 안 보인다.
→ §4 packing 검증은 decode에 의존하지 말고 `cu_seqlens` / `position_ids`를 직접 assert해야 한다.

**심각도 눈금**: template mismatch(잘못된 stop token → 무한 생성)는 5–20pt 급 재앙,
BOS 처리는 2급 버그 — 실재하고 silent하지만 절벽은 아니다. 같은 칸에 놓지 말 것.

관련: [[ch-32]] (reasoning SFT — `<think>` 스코프), [[ch-36]] (packed SFT lab — masking test가
위 검증 공백을 메워야 함), [[hf-alignment-handbook]]

## Q-정정 (2026-09 revision)

**Kernel.** 2026-09 revision에서 read.md를 primary source 기준으로 다시 썼다. 기존 entry 중 정정된 사실을 포함하는 것은 Q1이다. Q1과 이후 entry의 line reference(예: "read.md L134", "L87")는 git commit 4a72e54 시점의 read.md를 가리키며, 현재 read.md의 line 번호와 다르다.

Q1에서 정정되는 내용:

1. "template mismatch는 5–20pt 급 재앙"이라는 수치는 출처가 없다. 측정된 값은 Tülu 3 Table 13의 template 선택 실험이며, 다섯 template 사이의 평균 점수 차이는 1.4 point(51.6–53.0)이다 ([[tulu-3]] §4.3.1). 이 실험은 template 선택의 효과이고 training과 inference 사이 mismatch의 효과를 잰 것은 아니다.
2. "decode a packed batch and eyeball the delimiters"를 hf-alignment-handbook의 lesson("#1 silent bug")으로 인용한 부분은 handbook에서 확인되지 않았다 (alignment-handbook@1de1fc9의 `config_full.yaml`, `scripts/sft.py` 확인). 현재 read.md §6 Checks는 packed batch를 decode하는 검사 대신, 같은 conversation을 training template과 serving stack으로 render해서 token id를 비교하고 role별 label mask를 출력하는 검사를 권고한다.
3. "position-ID reset 누락"을 packing 버그로 든 부분의 이유가 바뀐다. block-diagonal mask가 올바르면 RoPE attention은 relative offset에만 의존하므로, 한 document 안의 일정한 offset은 attention을 바꾸지 않는다. position_ids가 중요한 경우는 kernel이 position_ids에서 `cu_seq_len`을 계산할 때(Kundu et al. §3.3), trained range를 넘는 position이 생길 때, learned absolute position embedding을 쓸 때이다 ([[sequence-packing-contract]]). 따라서 "decode 대신 `cu_seqlens`/`position_ids`를 직접 assert한다"는 결론은 유지되지만, 근거는 boundary 계산이다. boundary를 position_ids에서 계산하는 Kundu et al. 구현에서 position IDs 없이 packing하면 example끼리 attend하고, Mistral-7B(FLAN 20K)의 validation loss는 같은 35 optimizer step에서 1.221 대신 1.294였다 (Table 4). "trained range를 넘는 position"은 source가 측정한 결과가 아니라 이 course의 해석이다.
4. Q1이 전제한 old read.md §3의 template 표는 정정되었다. Mistral-7B-Instruct-v0.2에서 fine-tune한 FILM-7B는 ChatML이 아니라 `[INST] … [/INST]` template으로 학습했고 ([[in2-film]] App. D), Zephyr SFT config는 `<|system|>`/`<|user|>`/`<|assistant|>` + content + eos_token template을 쓴다 ([[chat-template-matrix]]).

관련: read.md "Corrections to the version you studied" 6, 14–16번 항목, [[ch-04]].
