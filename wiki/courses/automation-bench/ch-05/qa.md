<!-- qa for ch-05 — Task Anatomy and the Six Business Domains — see [[read]]
     Kernel only; full chains in read.md / discuss transcript. Append-only. -->

# ch-05 Q&A

## Q1. 비즈니스 도메인(task)은 어떻게 확보했고, "정해진 답"은 어떻게 측정하나?

**확보 = 긁은 게 아니라 합성 *저작*.** 실제 사용의 *모양(shape)* 만 추출(~2B 월간 task / 3.7M 회사 / 9,000+
통합에서 흔한 워크플로 패턴) → 가짜 엔티티로 손으로 재구성 + 함정 심음 → `domains/*/tasks.py`의 손으로 짠 생성자
dict. PII·raw 고객데이터 0. Zapier 실제 Agents 제품의 실패 피드백으로 hardening. **공개 안 함**: 생성 레시피(코드),
private 600+ task(리더보드용). 공개 606 public은 dict로 다 보임.

**"정해진 답"의 정밀화**: 고정된 건 단일 출력이 아니라 **assertion 집합**(must-pass + must-not-occur). 그 제약을
만족하는 *어떤* 최종 world든 통과(path-agnostic, 최종상태도 유일 아님).

**정답을 누가 정하나**: task가 합성 저작되니 **저작자가 seeded world(정책·함정)를 설계하면서 동시에 의도한 결과를
assertion으로 박아넣음** — 정답은 사후 라벨링이 아니라 저작 시점에 *구성*됨. (inbox에 정책 심는 사람 = 그 정책
지켰는지 검사하는 assertion 쓰는 사람.)

**측정**: episode 끝나고 `AssertionRegistry`가 각 assertion을 최종 world에 대해 순수 Python으로 검사 →
`partial_credit = passed/total`, `task_completed_correctly`는 전부 통과 시 1. LLM judge 없음.

**핵심 거래**: 합성이라서 정답을 *정의*할 수 있고(real 긁었으면 ground truth 라벨이 없어 불가능), 정답이
assertion이라 *측정*할 수 있음. 대가 = sim2real gap(높은 점수는 필요조건이지 충분조건 아님). → ch-09/ch-10,
그리고 학습자 본인 sales-AI eval 문제로 이어짐. ([[automationbench-results]], [[benchmark-comparison]])
