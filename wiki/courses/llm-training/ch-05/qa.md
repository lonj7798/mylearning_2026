<!-- chapter: ch-05
     companion to: [[read]]
     kind: qa
     track: foundations
-->

# ch-05 — Reading Q&A

## Q1. `TP 8 / CP 1 / PP 16 / DP 64`가 뭔가?

**Kernel.** 8,192 GPU를 네 축으로 동시에 쪼갠 것. TP 8×PP 16×CP 1=128 GPU가 모여야 405B 모델 한 벌이 완성되고, 이 128-GPU 단위를 DP 64로 복제. `world_size = TP×PP×CP×DP = 8,192`. see [[read]] §1, §2.2.

## Q2. Rank가 뭔가?

**Kernel.** 분산 학습에서 개별 프로세스(보통 GPU 1:1)를 식별하는 번호. "랭크당 batch"(§2.2 "Batch size/DP" 열)는 DP rank 하나가 한 step에서 처리하는 sequence 수. `global_batch_tokens = DP × sequences_per_DP_group × sequence_length`. rank는 DP 전용 용어가 아니라 어느 축(TP/CP/PP/DP)이든 "그 축 위의 한 GPU/프로세스"를 가리키는 범용 용어. see [[read]] §1, §2.2, §2.3.

## Q3. MFU가 뭔가?

**Kernel.** Model FLOPs Utilization = 실제 달성 TFLOPs/GPU ÷ 하드웨어 peak TFLOPs. Recipe 표 숫자(430/0.43≈1,000, 400/0.41≈976, 380/0.38≈1,000)로 역산하면 H100 BF16 peak(~989 TFLOPs)와 일치 — 이 장의 run들은 H100 클러스터. ch-05 본문에 직접 정의된 공식은 없음(일반 LLM 학습 용어를 보충 설명). see [[read]] §2.2 Table 4.

## Q4. "Stage"가 DP의 데이터 단위인가?

**Kernel.** 아니다, 두 가지 무관한 뜻이 이 장에 섞여 있다. (1) PP의 stage = 모델 layer를 나눈 조각(§1) — 데이터가 아니라 모델을 나누는 축. (2) OLMo 3의 training stage = pretrain/mid-train/long-context 같은 시간 순서상 커리큘럼 단계(§2.3) — parallelism과 무관. DP가 나누는 데이터 단위는 stage가 아니라 "sequences per DP group"(Q2). see [[read]] §1, §2.3.

## Q5. PP에서 stage 1이 마지막 layer인가(backprop 때문에)?

**Kernel.** 아니다. Stage 번호는 forward pass 방향(입력→stage1→...→stage16→출력) 기준으로 고정. Backward는 같은 stage들을 반대 순서(16→1)로 지나갈 뿐, 번호가 재배열되지 않는다. 이 방향성 차이가 바로 PP bubble의 원인: stage 1 GPU는 forward를 제일 먼저 끝내지만, 그 gradient는 stage 16부터 15개를 거쳐야 돌아오므로 그동안 대기하거나 다음 micro-batch를 처리해야 함. see [[read]] §1, Recipe(bubble ratio row).

## Q6. Stage마다 순차적으로 weight를 load/unload하면 안 되나?

**Kernel.** 두 가지 비용이 실재한다: (1) 매 stage 전환마다 weight+gradient+Adam moment(m,v) — 최소 4배 데이터 — 를 옮겨야 함. (2) 더 근본적으로, 물리적으로 분리된 16개 GPU 그룹이 각자 자기 stage weight를 상주시켜야 여러 micro-batch를 동시에 서로 다른 stage에서 처리하는 동시성(pipelining)이 성립한다. 순차 load 방식은 이 동시성을 완전히 제거해, 항상 최대 1개 micro-batch만 전체 모델 안에서 처리 중이게 됨 — bubble보다 훨씬 나쁨. see [[read]] §1 (Placement), Recipe (interleaved schedule row).

## Q7. AllGather+ReduceScatter(sharded) / AllReduce(replicated) / all-to-all이 뭔가?

**Kernel.**
- AllReduce: 각 rank가 다른 값을 갖고 있다가, 합산 후 **모두가 같은 결과**를 받음. Replicated DP의 gradient 동기화.
- AllGather: 각 rank가 다른 조각을 갖고 있다가, **모두가 전체 조각 모음**을 받음. Sharded DP(FSDP)에서 계산 직전 parameter 복원.
- ReduceScatter: 각 rank가 전체 벡터를 갖고 있다가, 합산과 동시에 쪼개져서 **각자 자기 몫만** 받음. Sharded DP의 gradient 동기화.
- All-to-all: 각 rank가 서로 다른 목적지용 조각을 갖고 있다가, **각자 자기에게 온 조각들만** 모아 받음(일대일 매핑). EP의 token dispatch/combine.
see [[read]] §1, §5.

## Q8. Sharding이 뭔가?

**Kernel.** 큰 데이터(parameter든 corpus든)를 여러 조각으로 쪼개 여러 rank에 나눠 저장 — 어느 rank도 전체를 안 갖고 자기 shard만 가짐. 반대는 replication(전체 복제). 두 가지 무관한 쓰임: (1) parameter sharding — §1의 sharding factor F (F=1 replication, F=W full sharding, 1<F<W hybrid). (2) data sharding — §3.2의 corpus를 파일 단위로 나눈 shard, DP rank가 자기 몫만 읽음. see [[read]] §1, §3.2.

## Q9. 트레이닝을 resume 가능하게 설계해야 하고, 중간에 오류가 날 수 있다는 관찰

**Kernel.** 맞다. 다만 이 장(§3.3)의 핵심은 "resume이 가능한가"가 아니라 "resume이 정확한가"다. OLMo 3의 `load_optim_state=True` + `load_trainer_state=False`는 stage 전환(다른 mixture)에서는 의도된 설정이지만, 같은 stage 재시작에 그대로 쓰면 optimizer state는 이어지고 data loader 위치는 instance 0부터 재시작 — **에러 없이, loss curve도 부드럽게** corpus 뒷부분을 영원히 못 보게 됨. see [[read]] §3.3.

## Q10. TFLOPs/MFU를 알면 하드웨어 스펙이 달라도 최대 성능을 뽑아내게 설계할 수 있나?

**Kernel.** 두 가지를 구분해야 한다. (1) Predictive 설계: §4.2 ring attention의 `c=F/B, s=6c` 공식처럼, FLOPS·bandwidth만 대입하면 실행 전에 임계값(최소 local sequence)을 미리 계산 가능 — 실제로 A100 NVLink(~6,200)와 InfiniBand(~149,500)에서 24배 차이. (2) MFU는 diagnostic — 돌려본 뒤 하드웨어 대비 효율을 알려줄 뿐, 원인은 별도로 추적해야 함(§2.2의 43%→41% 하락 원인을 rank당 batch 감소로 추적한 것처럼). see [[read]] §2.2, §4.2 Table 2.
