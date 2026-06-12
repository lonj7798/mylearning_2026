<!-- qa for ch-03 — Tool Discovery: Search + a From-Scratch BM25 — see [[read]]
     Index of clarifying questions / critiques raised while reading. Kernel only;
     full chains in read.md / discuss transcript. Append-only. -->

# ch-03 Q&A

## Q1. `search_tools`가 검색에 실패하면(매칭 0) 모델은 뭘 받아?

**빈 배열 `[]`.** 메시지도 힌트도 없음. 경로: `BM25Scorer.top_k`의 `s > 0` 필터가 무매칭이면 빈 리스트 →
`bm25()` 빈 리스트 → `search_tools`가 `json.dumps([])` = `"[]"`. `"no available tool"`도 `"재검색해봐"`도
**안 줌** → 모델이 *스스로* "비었다 = 내 검색어가 나빴다"를 진단하고 어휘를 바꿔 재검색해야 함(그 자발성이 측정 대상).

두 부작용:
- **부분 매칭이면 빈 게 아니라 *엉뚱한* tool이 나옴**(`s>0`이면 점수 낮아도 포함) → 헛 tool을 실행할 위험.
- **점수가 결과에 안 들어감**(`bm25()`는 name/description/parameters만 반환, BM25 score 제외) → 모델은 "강한
  매치"와 "간신히 걸린 매치"를 출력만으로 구분 못 함 → **false confidence**로 직결.

대조: `execute_tool`은 없는 이름으로 부르면 `"Unknown tool: ... Use search_tools..."` **힌트를 줌**. 검색 실패엔
힌트 없음 — 두 실패의 처리가 다름.

## Q2. BM25가 동의어를 놓치지 않나? (`modify_event` 있는데 `update_schedule`로 검색)

맞다 — BM25는 **글자(토큰) 매칭**이지 의미 매칭이 아님. 공유 토큰 0이면 점수 0 → 못 찾음. **완화책**: 색인 대상이
이름만이 아니라 **docstring + 파라미터 설명 전체**라 매칭 표면이 넓음(`reschedule`/`update`/`calendar`가
docstring에 있으면 잡힘). **의도된 선택인 이유**: (1) 현실성 — 실제 retriever(Zapier 제품)도 키워드식, oracle
의미검색은 배포 성능을 과대평가; (2) 결정성 — embedding은 외부의존·비용·비결정성을 재도입(ch-02가 피한 것);
(3) 분리가능 — discovery를 안 재고 싶으면 `limited_zapier`. **데이터**: discovery tax가 +1.5~2pt뿐이라 이론적
한계지만 측정 영향은 작음 → 모델이 재구성을 꽤 잘함. ([[automationbench-results]])

## Q3. 검색 실패 시 행동은 prompt에 좌우되지 않나?

그렇다. 실제 `SYSTEM_PROMPT`엔 재구성 코칭이 **없고**, 오히려 `"avoid duplicate searches"`로 재검색을 살짝 말림.
"모델이 재구성할 것"은 prompt가 시킨 게 아니라 모델 자발성 → 모델마다·prompt마다 갈림(agent 벤치마크는 prompt
바꾸면 ±10pt 출렁이는 게 알려진 현상). 벤치마크는 **prompt를 고정**해 모델 간 비교를 공정하게 만들지만, 절대 숫자는
"이 모델 × 이 prompt"지 본질 능력이 아님 → 다른 scaffold면 점수가 흔들림. **점수 = 고정 조건에서의 측정값**
(ch-01의 sim2real: 필요조건이지 충분조건 아님).
