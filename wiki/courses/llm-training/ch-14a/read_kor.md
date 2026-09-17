# 14a장 — 사전학습 레시피 비교

레시피 장부에는 모델 크기, 총 토큰, tokens per parameter, 글로벌 배치(토큰 수), peak·final learning rate, warmup, decay 모양, 단계별 데이터 혼합을 기록하고 공개 설정과 로그로 검증한다. 같은 모델이라도 배치와 학습률 스케줄이 다르면 업데이트 수와 최종 손실이 달라진다.

각 단계의 혼합 비율은 실제 토큰 점유율로 환산한다. 긴 문맥 단계는 긴 샘플을 upsample할 수 있으므로 long:short 비율과 길이별 토큰 수를 함께 기록한다. 사전학습 distillation을 쓰면 teacher의 분포와 추가 비용, 학생 모델의 held-out 손실을 레시피에 포함한다.

## Recipe

모든 설정을 한 표에 모으고 토큰·배치 단위를 통일한다. 단계별 체크포인트를 저장해 손실과 미공개 능력 프로필을 비교하고, 재개 시 optimizer·scheduler·dataloader 상태도 보존한다.
