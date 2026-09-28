<!-- chapter: ch-09
     companion to: [[read]]
     kind: qa
     track: pretraining
-->

# ch-09 — Reading Q&A

## Q1. "CC-100-trained model보다 bytes당 bit가 낮았다"는 무슨 말인가? (2026-09-28)

**Kernel.** BPB(bits per byte) = 모델이 텍스트를 byte당 평균 몇 bit로 표현하는가 = 음의 log-likelihood를 byte로 정규화한 값(`BPB = (L_T/L_B)·ℓ/ln2`, §2). 낮을수록 예측이 정확. byte 단위 정규화는 tokenizer와 무관하게 비교하기 위함. Worked example: L_T/L_B=0.29335, ℓ=2.0 nats → BPB 0.846. (256가지 byte를 무작위로 찍으면 log2 256 = 8 bit — 이 course의 일반 상식 보충.) 실험: The Pile / CC-100(영어; Wikipedia 학습 LM의 perplexity filter 사용) / raw Common Crawl로 각각 같은 구조의 1.3B 모델을 약 40GB로 맞춰 학습하고 Pile의 22개 component test set에서 각각 평가 — Pile 모델이 22개 전부에서 BPB가 가장 낮음(Table 4). 웹 텍스트(Pile-CC)는 0.9989 vs 1.0873(차이 0.09), 비웹 소스는 1 bit 이상 차이(ArXiv 0.7945 vs 1.8159, Github 0.5597 vs 1.6509, DM Mathematics 1.5206 vs 3.1774). 한계(§2): Pile 자신의 test set이라 Pile 모델은 in-distribution, CC 모델은 out-of-distribution이므로 "어느 코퍼스가 어느 domain을 덮는가(coverage)"를 보여줄 뿐 일반 품질의 비교가 아님; 40GB 크기 통제는 Pile의 약 1/3 크기인 CC-100에 유리. see [[read]] §2.
