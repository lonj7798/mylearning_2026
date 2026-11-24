<!-- chapter: ch-04
     track: foundations
     kind: content
     title: Sequence Packing, Loss Masking, and Chat Templates
     deps: [ch-00]
     sources: [[sequence-packing]], [[fewer-truncations-best-fit-packing]], [[sequence-composition-intra-doc-masking]], [[packing-flash-attention-position-ids]], [[packing-analysis-sft]], [[loss-masking-prompt]], [[prompt-loss-weight-sift]], [[neftune]], [[prompt-format-sensitivity-formatspread]], [[mind-your-format-templates]], [[flan-collection]], [[hf-alignment-handbook]], [[trl-sft-packing-masking]], [[transformers-chat-templates]], [[open-instruct-sft-masking]], [[open-instruct-allenai-recipes]], [[llama-3]], [[llama-3-recipe]], [[prolong]], [[prolong-recipe]], [[longalign]], [[string-effective-context]], [[tulu-3]], [[llama-2-long]], [[openr1]], [[openr1-recipe]], [[deepseek-v3-recipe]], [[deepseek-v4-recipe]], [[tulu-1-how-far-can-camels-go]], [[nemotron-4-synthetic-recipe]]
     figures: figures/packed-attention.html
     revised: 2026-09 (generality revision)
-->

# Chapter 4 — Sequence Packing, Loss Masking, Chat Template

> **핵심 통찰.** Packing, loss mask, chat template은 token이 어느 token에 attention할 수 있는지, 어느 token이 gradient를 받는지, 어느 token ID가 conversation 구조를 표시하는지를 정한다. 이 선택들은 어느 것도 error를 내지 않지만 각각 breadth에 측정된 효과가 있다. Truncation을 피한 packing은 13B text model의 reading-comprehension/context-following score를 올리고 7B code model undefined-name error를 최대 58.3% 줄였다([[fewer-truncations-best-fit-packing]]). Intra-document attention은 1.3B에서 in-context-learning accuracy를 올렸고([[sequence-composition-intra-doc-masking]]), instruction loss 추가는 short-output 또는 작은 SFT set의 overfitting을 줄였다([[loss-masking-prompt]], [[prompt-loss-weight-sift]]). Model은 evaluation에서도 format에 민감하며 동등한 prompt format 사이 accuracy spread가 최대 76점이다([[prompt-format-sensitivity-formatspread]]).
>
> **지침.** Example이 sequence length보다 짧으면 best-fit algorithm과 intra-document attention으로 pack하라. Padding은 제거하면서 example을 truncate하지 않고 [[fewer-truncations-best-fit-packing]]에서 평가한 score를 유의하게 낮추지 않았기 때문이다. Data가 short-completion(completion이 prompt보다 짧음)이거나 작으면(수천 example), prompt-loss weight를 0으로 고정하지 말고 tuned hyperparameter로 다뤄라. [[prompt-loss-weight-sift]]가 이 data에서 optimum 0.242를 fit했고 [[loss-masking-prompt]]가 instruction loss에서 overfitting 감소를 측정했기 때문이다. 그 외에는 Llama 3/Tülu처럼 prompt token을 mask하고 assistant end-of-turn token이 loss mask 안에 있는지 확인한다. Model이 여러 training stage를 지나거나 별도 tokenizer configuration에서 serving되면, 매 stage 전 training/preference/RL/serving template로 같은 conversation을 render해 token ID를 비교한다. OLMo 3 7B Think의 DPO/RL stage가 “a minor miscommunication”으로 SFT stage와 약간 다른 template을 썼고 차이는 released model에 있기 때문이다([[open-instruct-allenai-recipes]]). Model/checkpoint를 비교할 때는 여러 prompt format에서 평가한다. [[prompt-format-sensitivity-formatspread]]에서 다른 format 아래 ranking이 뒤집혔기 때문이다.

## 범용 모델에 이 장이 중요한 이유

Training run은 token ID, attention pattern, token별 loss weight를 소비한다. 이 장은 raw document/conversation에서 그것들을 만드는 세 단계다. Pipeline의 모든 stage에 적용된다. Pre-training/long-context continued pre-training의 packing/document masking(ch-32b), SFT loss mask/template(ch-30), preference optimization formatting-token mask(ch-39), agentic RL observation mask(ch-45b), evaluation format sensitivity(ch-47a)다.

문제는 측정 가능하다. Cross-document attention은 관련 없는 document content를 각 token context에 넣는다. Truncation은 뒤 token이 의존하는 context를 제거한다. Prompt mask는 short-completion data에서 loss signal 대부분을 제거하며, 인용 연구는 이를 overfitting과 연결한다. Template mismatch는 turn 시작/끝 ID를 바꾸므로 trained model이 server가 기대하는 stop token을 내지 않을 수 있다. 범용 model에서 각 경우 질문은 training mixture에 없던 task/format에서 choice가 behavior를 개선하는지이며, ch-00이 측정 방법을 정의한다.

## §1 Packing: example을 truncate하지 않고 padding 제거

**정의.** Packing은 여러 variable-length example을 fixed-length row 하나에 넣어 padding token이 거의 없거나 없게 한다. Padding token은 attention/loss에서 제외되는 filler ID다.

**문제.** Batch에서 가장 긴 example에 맞춘 padding은 compute를 낭비한다. Sequence length 512의 Wikipedia BERT pre-training에서 token 50%가 padding이었고, length 128 GLUE CoLA task에서는 89%였다([[sequence-packing]], abstract; chat data가 아닌 BERT data).

**메커니즘.** Common layout은 세 가지다.

1. Padding: row당 example 하나, batch에서 가장 긴 example까지 pad.
2. Concatenate-and-split(“wrapped”): 모든 example을 이어 붙여 매 \(L\) token에서 자른다. Padding은 없지만 cut을 넘는 example은 관련 없는 row로 split된다.
3. Bin packing: 각 example(\(L\)보다 길면 chunk)을 item으로 보고 split 없이 capacity \(L\) row에 배정한다. First-fit decreasing(FFD)은 item을 length순 sort해 space가 있는 첫 row에 둔다. Best-fit decreasing(BFD)은 남은 space가 가장 작지만 충분한 row에 둔다. Krell et al. shortest-pack-first histogram packing(SPFHP)은 worst-fit을 써서 각 length bin을 남은 space가 가장 많은 pack에 둔다([[sequence-packing]], §3.1.1).

**수식.**

```
EFF = (Σ_i L_i) / (R · L_max)          potential speed-up ≈ EFF_packed / EFF_padded
```

`L_i`는 example \(i\) length, `R`은 생성 row 수, `L_max`는 row length다. EFF는 non-padding position 비율인 packing efficiency다.

**Worked example.** Batch example length가 100, 300, 600, 1,000 token이라고 하자. 1,000까지 padding하면 4 row, 4,000 position에 real token 2,000이므로 \(\mathrm{EFF}=0.50\)이다. 2,048 row로 BFD하면 넷을 한 row에 넣어 \(\mathrm{EFF}=2000/2048=0.977\), padding position은 48개다.

이제 [[fewer-truncations-best-fit-packing]] Figure 1 example을 보자. Document length 14, 7, 5, 2, 3 token, \(L=8\)이다. Concatenate-and-split은 [0,8), [8,16), [16,24), [24,31) row를 만든다. 14/7/5-token document가 모두 cut을 넘으므로 5개 중 3개가 truncate된다. BFD는 먼저 14-token document를 8/6 chunk로 split하고 descending order로 배치한다. 8 → row A(0 left), 7 → row B(1 left), 6 → row C(2 left), 5 → row D(3 left), 3 → row D(가장 작은 충분 space, 0 left), 2 → row C(0 left). 두 layout은 row 4개를 쓰지만 BFD는 \(L\)보다 긴 document만 truncate한다.

**근거.**

- Efficiency: BERT Wikipedia data에서 packing 없이는 50.0%, pack당 sequence 최대 3 SPFHP는 89.4%, NNLSHP(non-negative least squares histogram packing)는 99.7% efficiency였고 NNLSHP IPU estimated speed-up은 1.913이었다. Phase-2 training measured total speed-up은 2×를 넘었다([[sequence-packing]], Table 1, §4.2, Fig. 3). **결과(단일 연구).**
- Scale compactness: RefinedWeb 2,048 token에서 BFD는 concatenation보다 row를 0.0024% 더했고, Stack 2,048 token에서는 0.0028% 더했다. Optimized BFD는 1B document에 10,816 s, FFD는 26,354 s였다([[fewer-truncations-best-fit-packing]], Tables 1-2).
- Capability: 500B RefinedWeb token의 13B model(2k/8k context), 300B Stack token의 7B code model pre-training에서 BFD packing은 reading-comprehension average를 46.71→48.92(2k), 47.13→49.03(8k), NQ-Swap context following을 45.62→51.03(2k)으로 올렸고, generated code undefined-name error를 최대 58.3% 줄였다. Statistically significant decrease는 어떤 task에도 없었다([[fewer-truncations-best-fit-packing]], §4, Tables 4-5). **결과(단일 연구).**
- Framework default: aa89588 TRL `SFTConfig`는 `packing=False`, `packing_strategy="bfd"`이며, `"bfd"`는 `max_length`보다 긴 example을 truncate, `"bfd_split"`은 split, `"wrapped"`는 mid-sequence cut한다([[trl-sft-packing-masking]], `trl/trainer/sft_config.py` L77-82; `trl/data_utils.py` L859-866).

**조건과 한계.** Packing은 throughput보다 많은 것을 바꾼다. Fixed example/epoch에서는 offline packing이 optimizer step 수를 줄인다. 20K FLAN subset의 Mistral-7B 1 epoch은 offline packing(585 training row) validation loss 1.284, padding(19,961 row) 1.129이었고, minibatch 내 packing은 step count를 유지해 1.127이었다([[packing-flash-attention-position-ids]], Table 2). LLaMA-3-8B/70B SFT에서는 greedy packing이 8개 model-dataset setting 모두에서 padding average보다 높고 random packing은 8개 중 2개에서 낮았으며, padding에서 보인 linear batch-size–learning-rate relation은 packing에서 성립하지 않았다([[packing-analysis-sft]], Table 3, §5.3). Open R1은 C++ competitive-programming R1 reasoning trace로 Qwen2.5-Coder Instruct model을 fine-tune했고 ablate한 모든 dataset에서 packing이 LiveCodeBench(Python)에서 “consistently worse”임을 발견했다. 저자 가설은 긴 trace가 chunk boundary에서 clip되어 question/answer가 분리된 것이며 C++-to-Python mismatch가 gap을 키웠다고 의심한다([[openr1]], Update #3 Lesson 1). 그 packing implementation은 sample이 chunk boundary를 겹쳐 넘게 했으며 BFD가 아니라 concatenate-and-split layout이다.

**범용 모델에 대한 함의.** 측정된 benefit은 example을 온전하게 보존하는 데서 나온다. Corpus에 \(L\)보다 긴 document/trace가 있다면 tail을 버리거나 split하는 packing strategy는 context-following/long-generation task가 필요한 long-range dependency를 제거한다.

## §2 문서 간 attention: causal masking 대 intra-document masking

**정의.** Packed row 전체의 causal masking에서는 각 token이 earlier document를 포함한 row의 모든 앞 token에 attention한다. Intra-document causal masking에서는 각 token이 자기 document 앞 token에만 attention하며 attention mask는 block-diagonal이다. 두 likelihood는 다음이다.

```
causal:          P(C) = Π_{t=1..|C|} P(x_t | x_1, …, x_{t−1})
intra-document:  P(C) = Π_{i=1..n} Π_{j=1..|d_i|} P(d_ij | d_i1, …, d_i(j−1))
```

`C`는 길이 `|C|` packed row, `x_t`는 t번째 token, `d_i`는 row의 \(n\) document 중 i번째, `d_ij`는 document i의 j번째 token이다([[sequence-composition-intra-doc-masking]], §2.2).

**문제.** 관련 없는 document token이 모든 attention head softmax normalization에 들어간다. Model은 관련 없는 text를 condition으로 token을 예측하도록 학습된다.

**수치 예가 있는 메커니즘.** Query가 자기 document의 앞 token 둘에 attention score 2와 1을 가진다고 하자. Block-diagonal mask에서 weight는 \(e^2/(e^2+e^1)=0.731\), 0.269다. Mask가 없고 previous document token 하나도 score 2면 denominator는 \(e^2+e^1+e^2=17.50\)이며 weight는 0.422, 0.155, 0.422다. 관련 없는 token이 attention 42%를 받고 loss gradient가 그 token으로 흐른다.

**구현.** FlashAttention variable-length kernel은 cumulative boundary `cu_seqlens`(length 3,2,4이면 `[0, 3, 5, 9]`)를 받고 \(L\times L\) mask materialize 없이 block-diagonal causal attention을 계산한다. Hugging Face padding-free path에서 boundary는 `position_ids`에서 파생된다. `attention_mask`가 없으면 model attention code가 `position_ids`에서 `cu_seq_len`을 계산한다([[packing-flash-attention-position-ids]], §3.3). 따라서 TRL은 이 path에서 attention mask를 생략하고, example 하나의 last token이 다음 example first token을 예측하도록 학습하지 않게 각 example first token label도 제거한다([[trl-sft-packing-masking]], `sft_trainer.py` L485-486, L516).

```python
# trl/trainer/sft_trainer.py @ aa89588, L484-L516 (abridged)
# For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and
# compute wrong cu_seq_lens from the all-1s mask
...
output["labels"][output["position_ids"] == 0] = -100
```

Companion figure [figures/packed-attention.html](figures/packed-attention.html)는 §3 끝에서 설명하듯 수정 가능한 document length에 대한 이 mask를 그린다.

**근거.**

- SlimPajama token 150B로 1.3B model을 pre-train할 때 intra-document masking은 average perplexity를 9.172→8.410(2K context), 9.065→8.079(8K)로 낮췄다. 7개 classification task average in-context-learning accuracy는 63.54→70.52(2K), 62.43→71.23(8K)로, closed-book QA exact match는 7.99→10.99(8K)로 올랐다. 저자 implementation runtime cost는 4%였다([[sequence-composition-intra-doc-masking]], Tables 1-3, §3.1).
- 13B 2k ablation에서 concatenation에 document mask를 더하면 perplexity는 9.64→9.53, reading comprehension은 46.71→47.92였다. Mask가 있는 BFD packing은 48.92에 도달했고, mask가 있는 concatenation은 summarization ROUGE-2를 12.16→11.79로 낮췄다([[fewer-truncations-best-fit-packing]], App. C.3, Table 10).
- Llama-3-8B long-context continued training(64K의 5B token)에서 cross-document masking은 mask 없음의 long-context average 53.6, short-task average 64.9에 대해 54.6, 65.5를 냈다([[prolong]], App. B.2, Table 20).
- Llama 3는 document mask를 썼고 standard pre-training에서는 “limited impact”, “very long sequences” continued pre-training에서는 중요하다고 수치 없이 보고한다([[llama-3]], §3.2).
- BERT에서 mask adjustment를 제거하면 recovery 없이 loss/accuracy가 “worsen drastically”했다([[sequence-packing]], §4.2.1).

상태: 1.3B–13B pre-training 및 long-context training에서 **복제됨**([[sequence-composition-intra-doc-masking]], [[fewer-truncations-best-fit-packing]], [[prolong]]). Lab마다 관행은 다르다. DeepSeek-V3는 cross-sample masking 없이 pre-training document를 pack했지만 SFT에는 sample masking을 썼고, DeepSeek-V4는 pre-training에 sample-level masking을 추가했다([[deepseek-v3-recipe]], §4.1, §5.1; [[deepseek-v4-recipe]], §4.1).

**조건과 한계.** 위 연구는 pre-training 또는 long-context continued training이며 verified source 중 decoder SFT에서 mask를 isolate한 것은 없다. [[packing-analysis-sft]]는 attention reset을 명시하지 않는다. [EOS] separator가 model이 인접 sample을 구분하게 한다고 말하며(§3.3.3), pack된 unrelated sample이 “fake” multi-turn conversation을 만든다고 주장한다. Filtered 200K single-turn OpenHermes 2.5 set을 packing한 LLaMA-3-8B에서는 MATH가 하락했고, data의 1/40–1/20에 multi-turn conversation을 추가하자 회복됐다. 같은 저자들은 internal 200K single-turn dataset에서는 few-shot benchmark decline이 없었다고 보고하며 차이를 data quality 탓으로 돌린다(§5.3). Decoder SFT에서 cross-contamination effect 크기는 **열린 질문**이다.

## §3 Packing 아래 position ID: 언제 reset이 중요한가

**정의.** Position ID는 positional encoding에 넣는 integer position이다. Packing 아래에서는 document마다 0에서 reset할 수 있고(`[0,1,2, 0,1, 0,1,2,3]`), row를 따라 계속할 수도 있다(`[0,1,2,3,4,5,6,7,8]`).

**RoPE의 메커니즘.** Rotary position embedding(RoPE)은 position에 비례하는 angle로 query/key vector를 rotate한다. Frequency \(\theta\) 하나에서,

```
⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩
```

`q`,`k`는 query/key vector, `R_m`은 angle \(m\cdot\theta\) rotation, `m`,`n`은 query/key token position이다. Attention logit은 offset \(n-m\)에만 의존한다.

**Worked example.** \(\theta=1\) rad, \(q=k=(1,0)\)이라고 하자. Position 0-2의 3-token document에서 last token이 first token에 attention하면 \(\cos(2-0)=-0.416\)이다. 같은 document를 position 5-7에 두어도 \(\cos(7-5)=-0.416\)이다. Block-diagonal mask에서는 허용 pair가 두 numbering에서 같은 offset을 가지므로 document 내부 attention은 변하지 않는다. [[fewer-truncations-best-fit-packing]]도 이를 이용한다. Cross-document attention을 mask하면 “thanks to the relative nature of rotary positional embeddings (RoPE), we do not adjust position ids”라고 말한다(App. C.1).

**Reset이 중요한 경우.**

1. Learned absolute position embedding: model은 position마다 vector를 lookup한다. Position adjustment 없는 packed BERT에서 MLM accuracy는 target 72.1% 대비 71.8%에서 stall했다([[sequence-packing]], §4.2.1).
2. `position_ids`에서 document boundary를 infer하는 framework: Hugging Face padding-free path에서 continuous position ID에는 0이 하나뿐이므로, 여기서 계산한 boundary는 `[0, total]`이 되고 kernel은 모든 document 사이에 attention한다([[packing-flash-attention-position-ids]] §3.3의 rule에서 도출). Verified source는 이 failure의 loss effect를 isolate하지 않는다. [[packing-flash-attention-position-ids]] Table 2에서 offline packing 두 variant는 모두 padding보다 높은 validation loss였다(Mistral-7B: position ID 없음 1.306, 있음 1.284, padded 1.129). 두 packed variant는 row count도 다르다(Mistral-7B 2,294 대 585 row). Variant gap은 model에 따라 sign도 바뀐다(Llama-2: 1.579 대 1.578; Granite: 1.740 대 1.767). 이 pattern은 offline packing의 더 적은 optimizer step이 지배한다는 저자 설명과 일관된다.

**Position frequency.** [[string-effective-context]]는 relative distance \(i\)가 학습되는 빈도를 다음으로 정의한다.

```
f(i) = Σ_{s ∈ C} max(|s| − i, 0),   0 ≤ i < L
```

`C`는 sequence set으로 본 training corpus, `s`는 sequence 하나, `|s|`는 length, `i`는 relative distance, `L`은 training length다(§2.2, Eq. 2). 논문은 formula를 training sequence에 적용한다. 이 장은 attention 허용 span에 적용한다. Document mask가 있으면 span은 document, 없으면 whole row다. 3/5 token document가 든 8-token row 하나에서 masked count는 \(f=[8,6,4,2,1,0,0,0]\), unmasked count는 \([8,7,6,5,4,3,2,1]\)이다. Distance \(\ge3\)에서 trained pair 비율은 \(15/36=42\%\)에서 \(3/21=14\%\)로 하락한다. 논문은 SlimPajama \(L=2{,}048\)에서 distance \(\ge1{,}536\)이 position index의 5% 미만이며, 드물게 학습된 distance를 training length보다 짧은 effective context length와 연결한다고 보고한다(§2.2). Document 자체가 길지 않으면 document masking이 long-distance training을 줄인다는 함의는 이 장 Eq. 2에서 도출한 **해석**이다. ProLong long-context data는 길이 64K 이상 single document를 썼고 short data만 pack했다([[prolong-recipe]], §6.1, App. A.2).

**Figure.** [figures/packed-attention.html](figures/packed-attention.html)는 reader가 document length를 수정하고 document mask/position-ID numbering을 toggle하여 결과 `cu_seqlens`, padding-free framework가 `position_ids`에서 infer할 boundary, distance별 \(f(i)\), example별 loss share를 보게 한다.

## §4 Loss masking, prompt-loss weight, loss aggregation

**정의.** Loss mask는 token label을 −100(PyTorch `ignore_index`)으로 두어 해당 token이 loss term에 기여하지 않게 한다. Instruction tuning(IT)은 completion에만 loss를 계산한다. Instruction modelling(IM)은 instruction token에도 계산하되 template token에는 계산하지 않는다([[loss-masking-prompt]], §3, Eqs. 2,4). Prompt-loss weight(PLW)는 prompt token loss에 0~1 value를 scale한다([[prompt-loss-weight-sift]], §2).

```
L_IT = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})
L_IM = − Σ_{t=1..m+n} 1(x_t ∉ T) · log P(x_t | x_1..x_{t−1})
L_PLW = − Σ_{t ∈ completion} log P(x_t | x_<t)  −  w · Σ_{t ∈ prompt} log P(x_t | x_<t)
```

`I_1..I_m`은 \(m\) instruction(prompt) token, `C_1..C_n`은 \(n\) completion token, `x`는 concatenate sequence, `T`는 `<|user|>`, `<|assistant|>` 같은 template token set, `1(·)`은 indicator, \(w\in[0,1]\)은 prompt-loss weight다. Token count에 의한 \(L_{PLW}\) normalization은 implementation-specific이다.

**문제.** Completion이 짧으면 prompt mask는 example당 loss token을 적게 남기고 model은 이를 memorize할 수 있다.

**Worked example.** AlpacaDataShort의 mean prompt는 179.27 token(16.93 instruction + 162.34 input), mean completion은 14.62 token이다([[prompt-loss-weight-sift]], Table 1). Weighted loss token에서 prompt share는 \(w\cdot179.27/(w\cdot179.27+14.62)\)다. \(w=1\)에서 0.925, \(w=0.242\)에서 0.748, \(w=0.01\)에서 0.109, \(w=0\)에서 0이다.

**근거.**

- Shi et al.(주로 LLaMA-2-7B, LLaMA-2-13B/OPT-6.7B 점검; instruction dataset 1,030–326,181 example; 18 NLP task + MT-Bench/AlpacaEval)은 IM이 “in many scenarios”에서 NLP/open-ended benchmark 모두를 개선하며 instruction이 output보다 길고 example이 적을수록 가장 크다고 보고한다(§4.2, Fig. 2). Table 1에서 Alpagasus Dolly 9k NLP mean 45.54→48.00, AlpacaEval 1.0 21.54→30.77; Less MMLU Chat AlpacaEval 1.0 4.42→9.78; LIMA NLP mean 48.79→49.60, AlpacaEval 1.0 33.06→32.94다. 저자는 IM을 standard fine-tuning의 대체로 “not proposing”한다고 말한다(abstract).
- 같은 논문의 overfitting 측정: LIMA output mean training loss는 IM 1.45, IT 1.37이고 10% Tulu V2 sample test loss는 IM 1.17, IT 1.32였다. Training response에 대한 greedy output BLEU는 Table 2의 7 dataset 모두 IM이 낮았다(§4.3). IM 대신 IT에 base model KL penalty를 더하면 LIMA AlpacaEval 2.0은 2.58→0.06, NLP mean은 48.79→49.26이었다(Table 3).
- Huerta-Enochian & Ko(LLaMA 1 7B/LLaMA 2 7B, PLW 10 value, Alpaca variant 3개, benchmark 13개)는 short-completion data에서만 PLW/성능의 statistically significant negative quadratic relation과 fitted optimum \(\lambda=0.242\)를 발견했다. PLW 0.01–0.5는 multiple-choice/short-generation benchmark에서, 1 근처는 long-generation judge benchmark에서 좋았다. Long-completion dataset 둘은 relation이 statistically significant하지 않았다(\(p=0.237,0.0861\))(§5.1-5.2, Table 3). 저자는 training-loss stability가 driver가 아니라고 잠정 결론내고, small non-zero PLW의 pretrained model에 더 가까운 weight와 larger PLW의 lower memorization을 지목한다(§5.3).
- Llama 2 Long Chat은 long input prompt에 language-modelling loss를 더했다. Self-instruct long data에서 QuALITY는 input loss 없이 59.3에서 input loss 포함 77.3, Qasper는 35.7→38.9였다([[llama-2-long]], Table 9, model size 미보고).

상태: short-completion 또는 long-input data의 direction은 서로 다른 setting에서 [[loss-masking-prompt]], [[prompt-loss-weight-sift]], [[llama-2-long]]에 걸쳐 **복제됨**이다. Tulu V2처럼 large long-completion mixture(instruction/output length ratio 약 0.5)에서는 reported gain이 작았고([[loss-masking-prompt]], §4.2), Huerta-Enochian & Ko는 long-completion data에 statistically significant relation이 없었다. 이 장의 released recipe는 ablation 없이 prompt를 mask한다. Llama 3 SFT([[llama-3]], §4.1.3), Tülu 1([[tulu-1-how-far-can-camels-go]], §3.2), Nemotron-4-340B-Instruct([[nemotron-4-synthetic-recipe]], §3.3.1), ProLong SFT([[prolong-recipe]], App. A.3)다.

**피해야 할 구현 오류.** Shape [B,T] batch tensor에서 `labels[:prompt_len] = -100`은 token position이 아니라 첫 `prompt_len`개 row를 mask한다. Example별 형식은 `labels[b, :prompt_len[b]] = -100`이며, 이어서 `logits[:, :-1]`와 `labels[:, 1:]`의 one-position shift를 적용한다.

**Multi-turn conversation.** 세 rule이 있다. All-assistant masking은 one pass에서 모든 assistant turn을 학습한다. Open-instruct `sft_tulu_tokenize_and_truncate_v1`은 non-assistant message 전체를 mask하고, 별도 `last_turn_tulu_tokenize_and_truncate_v1`은 final message만 학습한다([[open-instruct-sft-masking]], `dataset_transformation.py` L1176-1225). Per-turn unrolling은 assistant turn당 example 하나를 만든다.

Worked example: \(u_1=40,a_1=50,u_2=30,a_2=80,u_3=20,a_3=120\) token이다. All-assistant masking은 340 token 처리/250 학습, last-turn masking은 340 처리/120 학습, unrolling은 \(90+200+340=630\) 처리/같은 250 학습이다. Causal attention에서 각 \(a_1\) token 예측은 earlier token만 의존하므로 unrolling은 all-assistant masking과 같은 token별 loss를 \(630/340=1.85\)배 compute로 내며 loss average 방식만 다르다. 이 equality는 뒤에 무엇이 오는지와 관계없이 earlier message를 동일하게 render하는 prefix-preserving template를 요구한다([[trl-sft-packing-masking]], `chat_template_utils.py` L829-831). Multi-turn task에서 어떤 rule이 더 잘 generalize하는지는 verified source에 보고되지 않았다(**열린 질문**).

**Framework default.** aa89588 TRL에서 `completion_only_loss=None`은 prompt-completion dataset completion에, language-modelling dataset full sequence에 loss를 계산한다. `assistant_only_loss=False`는 conversational dataset full sequence loss를 계산한다([[trl-sft-packing-masking]], `sft_config.py` L96-105). `assistant_only_loss=True`이면 assistant mask는 template `{% generation %}` marker에서 나온다. Tokenizer template에 marker가 없으면 TRL은 열거한 family(Llama 3, Qwen2.5, Qwen3 포함)에 patched training template를 넣고, 다른 template에는 `ValueError`를 낸다(`sft_trainer.py` L1251-1256; `chat_template_utils.py` L1032-1043, L1196-1199). Assistant token이 없는 example에는 error를 내며 assistant end-of-turn token이 mask 밖이면 “the model may not learn to stop”이라고 warn한다(`sft_trainer.py` L1258-1266, L1585-1591). Alignment Handbook Zephyr SFT config 1de1fc9는 `packing`, `assistant_only_loss` 둘 다 설정하지 않는다([[hf-alignment-handbook]], `recipes/zephyr-7b-beta/sft/config_full.yaml`). 따라서 위 TRL default에서는 packing 없이 모든 token을 학습한다(Handbook 명시가 아닌 도출). 같은 config의 `max_seq_length`는 aa89588 TRL `SFTConfig` field가 아니며 field는 `max_length`다. Effective setting은 installed TRL version에 의존한다.

**Loss aggregation.** Token-mean loss는 summed token loss를 batch target token 수로 나누며, sequence-mean은 example별 mean을 average한다. Tülu 3는 gradient accumulation/data parallelism이 batch를 split하면 Transformers default가 weight를 바꿈을 보인다.

```
one pass:      L = (l_1 + l_2) / (n_1 + n_2)
accumulation:  L = (l_1/n_1 + l_2/n_2) / 2
```

`l_k`는 example \(k\) summed token loss, `n_k`는 non-padding target token 수다([[tulu-3]], §4.3.2, Eqs. 1-2). Tülu 3는 sum loss를 쓰고 Tülu 2 mixture의 Llama 3.0에서 sum loss/LR 5e-6이 best라고 발견했다(Fig. 5). Packing은 third weighting을 더한다. Pack별 token mean이 동등하면 target token이 적은 pack example이 더 많은 weight를 받는다. LongAlign은 example \(i\) loss를 \(K/(N_i\cdot M)\)로 scale해 pack 안에서 sum한다.

```
L′ = (1/K) Σ_{k=1..K} Σ_{i ∈ pack k} L_i · K / (N_i · M) = (1/M) Σ_{i=1..M} L_i / N_i
```

`K`는 batch pack 수, `M`은 example 수, `N_i`는 example i target token, `L_i`는 summed loss다([[longalign]], §3.3, Eq. 4).

Worked example: pack 1은 A(\(N=10,L=20\)), C(\(N=20,L=30\)), pack 2는 B(\(N=90,L=90\))를 가진다. Per-pack token mean average는 \((50/30+90/90)/2=1.333\)이며 A/C/B weight는 0.167/0.333/0.500이다. Batch token mean은 \(140/120=1.167\), weight 0.083/0.167/0.750이다. LongAlign scaling은 \((20\cdot2/30+30\cdot2/60+90\cdot2/270)/2=1.5\)이며 각 \(1/3\) weight sequence mean \((2.0+1.5+1.0)/3\)과 같다. Krell et al.도 BERT per-sequence loss의 같은 per-pack weighting 문제를 설명한다([[sequence-packing]], §3.2.3). LongBench-Chat에서 loss weighting packing은 ChatGLM3-6B-64k에서 packing alone 5.76 대비 6.21, Llama-2-7B-64k에서 5.89 대비 6.10이었다. Single-document QA는 65.0→64.0, 61.7→60.8이었다([[longalign]], Table 3). **결과(단일 연구).**

## §5 NEFTune: 측정된 이득이 판정 모델 선호도인 정규화기

**정의와 수식.** NEFTune은 fine-tuning 동안에만 입력 embedding에 균등 잡음을 더한다.

```
X′ = X + (α / √(L·d)) · ε,   ε ~ Uniform(−1, 1)^{B×L×d}
```

여기서 `X`는 형상이 B×L×d인 embedding 출력이고, `B`는 batch size, `L`은 sequence length, `d`는 embedding dimension, `α`는 noise scale이다. Batch 안의 길이가 서로 다르면 scale은 sequence별로 계산한다([[neftune]], Algorithm 1 및 그 각주).

**근거.** LLaMA-2-7B의 AlpacaEval win rate(GPT-4 judge, Text-Davinci-003을 상대로)는 Alpaca에서 29.79에서 64.69로 상승했다. Evol-Instruct, ShareGPT, OpenPlatypus에서는 각각 70.34에서 79.60, 68.74에서 76.28, 62.00에서 70.61로 상승했다. ARC, HellaSwag, MMLU, TruthfulQA는 안정적으로 유지되었다([[neftune]], Table 1, Fig. 3). Training loss는 더 높았고 held-out loss는 약간 더 낮았다(Fig. 4). Alpaca에서 LLaMA-2 7B의 평균 출력 길이는 375.22자에서 1,061.89자로 늘었다(Table 4). GPT-4 judge를 쓴 LLaMA-1 Alpaca-7B에서 긴 답변을 요청하는 prompting은 48.01을, 250 token까지 end-of-sequence token을 막는 방법은 38.58을 기록했다. 이는 baseline 32.36 및 NEFTune 61.99와 비교한 값이다(Table 5). 따라서 길이만으로는 이 이득을 재현하지 못했다. [[loss-masking-prompt]]에서 NEFTune은 Less Tydiqa의 NLP mean을 48.21에서 47.47로 낮추는 한편 AlpacaEval 1.0을 5.12에서 8.35로 올렸다(Table 1). 측정된 이득은 judge-preference 이득이며, 네 개의 multiple-choice benchmark는 안정적으로 유지되었다(Fig. 3).

**Packing과의 상호작용.** Transformers는 hook을 다음과 같이 구현한다.

```python
# src/transformers/integrations/neftune.py @ 05e078a, L47-L50
if module.training:
    dims = torch.tensor(output.size(1) * output.size(2))
    mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
    output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
```

`output.size(1)`은 row length다([[transformers-chat-templates]]). α = 5, d = 4,096일 때 512-token example 하나는 scale (5/√(512·4,096)=0.00345)를 받는다. 이를 다른 일곱 example과 padding-free 4,096-token row 하나로 pack하면 (5/4,096=0.00122)를 받는다. 이는 Algorithm 1의 sequence별 값의 (1/√8=0.354)이며, 이 값은 코드와 algorithm footnote에서 도출된다.

## §6 Chat template: turn marker, generation prompt, 그리고 오류를 내지 않는 불일치

**정의.** Chat template은 tokenizer와 함께 저장되는 Jinja program으로, role/content message 목록을 각 turn의 시작과 끝을 표시하는 control token을 포함한 하나의 token sequence로 render한다. `add_generation_prompt=True`는 assistant turn을 여는 token을 덧붙인다([[transformers-chat-templates]], `chat_templating.md` L129-169).

**배포된 template의 설계 요소.**

- Alignment Handbook config의 Zephyr는 user에 `'<|user|>\n' + content + eos_token`, assistant에 `'<|assistant|>\n' + content + eos_token`을 쓰고, `'<|assistant|>'`를 generation prompt로 쓴다([[hf-alignment-handbook]], `config_full.yaml` L8).
- Llama 3는 “special header and termination tokens”를 사용한다. Header는 각 message의 출발지와 목적지(예: user 또는 tool execution을 위한 ipython)를 주고, termination token은 화자를 교대할 시점을 표시한다([[llama-3]], §4.1.1).
- Qwen3는 같은 assistant tool-call message가 마지막 message일 때에는 비어 있는 `<think>\n\n</think>` block을 넣어 render하고, 뒤에 tool message가 이어질 때에는 그 block 없이 render한다. Tool result는 `<tool_response>`로 user turn 내부에 render한다([[trl-sft-packing-masking]], `chat_template_utils.py` L1071-1077). 이러한 template은 prefix-preserving이 아니다. TRL의 patched Qwen3 training template은 두 경우 모두 빈 block을 render한다(L1078-1085).
- OLMo 3 32B Think는 `add_generation_prompt`에 `<think>` token을 넣은 상태로 배포되었고, OLMo 3.2 think template도 같은 방식으로 이를 추가한다([[open-instruct-allenai-recipes]], `docs/olmo3.md` L23, L29). Ai2 문서는 그 tokenization code가 “incorrectly masks the first <think> token as part of the prompt”한다고 보고한다. 그래서 OLMo 3.2+ Think SFT data는 `<think>`를 포함하지 않는 instruct template으로 tokenize하여 모델이 해당 token을 생성하도록 학습한다([[open-instruct-allenai-recipes]], `docs/olmo3.md` L26-28, L39). 이 mechanism은 masking code에서 확인할 수 있다. 각 masked prompt span의 끝은 `add_generation_prompt=True`로 prefix를 render하여 계산하므로, generation prompt 내부의 token은 모두 mask된다([[open-instruct-sft-masking]], `dataset_transformation.py` L1159-1171; 코드에서 도출).

**오류를 내지 않는 failure mode.**

1. Training과 serving 사이의 stop token이 다르다. Model은 한 id로 turn을 끝내지만 server는 다른 id를 기다린다.
2. Stage마다 template이 다르다. OLMo 3 7B Think의 DPO와 RL model은 “a minor miscommunication” 때문에 SFT와 “slightly different chat template”을 사용했으며, 이는 배포된 model에 반영되어 있다([[open-instruct-allenai-recipes]], `docs/olmo3.md` L22).
3. Special token이 중복된다. BOS 또는 EOS를 추가하는 tokenizer에서 `tokenize=False`로 render한 뒤 기본값 `add_special_tokens=True`로 tokenize하면 template이 이미 포함한 token을 중복한다([[transformers-chat-templates]], `chat_templating.md` L125-127).
4. End-of-turn token이 loss mask 밖에 있다(§4).
5. Generation prompt가 다르다. In-domain reasoning trace로 distillation한 Open R1 model은 coding query에 `<think>`를 냈지만, response를 `<think>`로 prefill하지 않으면 out-of-domain query에서는 원래 instruct behavior로 되돌아갔다. 저자는 release template에서 prefill을 권장한다([[openr1]], Update #3 Lesson 4).
6. 위 Qwen3 사례처럼 training과 inference에서 history를 서로 다르게 render한다.

**효과 크기에 관한 근거.** Llama 3.0을 쓴 intermediate Tülu SFT mixture에서 template choice의 average는 53.0(Tülu template, trailing newline을 EOS로 교체), 52.9(Zephyr), 52.8(Tülu 3, trailing newline 없음), 52.6(Tülu 2), 51.6(Llama 3 template)이었다([[tulu-3]], §4.3.1, Table 13). 비교를 위해, 최종 Tülu 3 SFT run은 서로 다른 setup에서 8B는 seed에 따라 59.8에서 60.1, 70B는 70.0에서 72.6까지 변했다(Table 14). Llama 3는 DPO response 둘 모두에서 header와 termination token을 mask했다. 저자는 이를 포함하면 “may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens”라고 관찰한다([[llama-3]], §4.1.4). Training-serving template mismatch의 수치는 verified source에 보고되지 않았다.

## §7 형식과 template 과적합

**정의.** Format sensitivity는 의미를 보존하는 prompt 형식들 사이에서 model score가 변하는 정도다. FormatSpread는 \(\max_i m(p_i,D)-\min_i m(p_i,D)\)인 spread를 보고한다. 여기서 `p_i`는 동등한 형식, `D`는 evaluation set, `m`은 metric이다([[prompt-format-sensitivity-formatspread]], §3.2).

**Worked example.** 동등한 형식 다섯 개의 accuracy가 0.62, 0.70, 0.55, 0.81, 0.66이라면 spread는 \(0.81-0.55=26\) points다. 한 형식만 보고하는 paper는 이 interval 안의 어느 값이든 보고할 수 있다.

**근거.**

- 53개 Super-NaturalInstructions task와 LLaMA-2 7B/13B/70B, Falcon-7B, Falcon-7B-Instruct, GPT-3.5에서 LLaMA-2-13B의 spread는 76 points에 이르렀다. Task당 10개 sample format으로 median spread는 7.5 points였고, 모든 LLaMA-2 setting에서 task의 20%는 최소 15 points spread를 보였다. 10개 형식만 sample했으므로 이는 lower bound다. More shots, larger model, instruction tuning 뒤에도 spread는 지속되었다([[prompt-format-sensitivity-formatspread]], §1, §4.2). LLaMA-2-13B와 70B의 model ranking은 다른 형식에서 최소 2 points 이상 뒤집힐 확률이 0.141이었다(§4.2).
- 21개 model과 4개 classification dataset에서 template 사이 standard deviation은 mean score의 35%에 이르렀다. 두 model의 top-10 template은 일부 model pair에서만 IoU 0.5 초과로 겹쳤고, instruction-tuned Llama 3와 Mistral model도 민감했다([[mind-your-format-templates]], §4.1, §5, App. L). 이는 [[prompt-format-sensitivity-formatspread]]에서 **복제됨**이다.
- Training-side 근거: Flan 2022 collection을 mixed zero-shot/few-shot template으로 fine-tune한 T5-XL은 held-in task에서 few-shot 74.8, MMLU에서 few-shot 52.4를 기록했다. Few-shot template을 제거하면 각각 62.2와 38.7이 되었고, zero-shot MMLU도 50.3에서 47.3으로 하락했다([[flan-collection]], Table 1). 같은 ablation은 zero-shot chain-of-thought score를 35.8에서 38.9로, zero-shot BBH score를 26.2에서 27.6으로 올렸다(Table 1). Mixing ratio 전반에서 10–90% few-shot template으로 학습한 경우는 하나의 prompt setting으로 학습한 경우보다 held-in과 MMLU score가 높았다(§3.2, Fig. 3). 이는 3B encoder-decoder의 **결과(단일 연구)**다. Decoder chat model SFT 중 chat template 또는 system prompt를 다양화하면 format spread가 줄어든다는 직접 근거는 verified source에 보고되지 않았다(**열린 질문**).

**함의.** 하나의 template에서 잰 score는 형식에 대한 distribution의 sample 하나다. §6의 Open R1 `<think>` 결과는 training-side 판본을 보인다. In-domain reasoning trace에서 배운 behavior는 response를 `<think>`로 prefill할 때만 out-of-domain prompt에서 나타났다.

## §8 장문 sequence: document masking, position id, length-aware loss

32K–512K token에서는 위 선택들이 length distribution과 상호작용한다.

1. **Document masking.** Llama 3는 매우 긴 sequence로 continued pre-training할 때 document mask가 중요하다고 발견했다([[llama-3]], §3.2). ProLong은 64K에서 mask로 long-context +1.0 point, short-task +0.6 point를 측정했다([[prolong]], Table 20). FSDP를 쓴 H100 GPU 8개에서 ProLong의 variable-length attention만으로는 throughput이 GPU당 2,770에서 2,780 tokens/s로 변했고, minibatch reordering을 결합한 variable-length attention은 3,095 tokens/s로 full attention보다 11.7% 증가했다([[prolong]], App. A.3, Table 15).
2. **Position id.** RoPE와 올바른 mask에서는 reset이 attention을 바꾸지 않는다(§3). Document mask가 있으면 long relative distance는 long document만으로 학습된다(§3 Eq. 2에서 도출). [[string-effective-context]]는 드물게 학습된 distance를 training length보다 짧은 effective length와 연결한다(§2.2, §3).
3. **Truncation.** TRL의 기본 `"bfd"` strategy는 `max_length`를 넘는 token을 버리고, `"bfd_split"`은 이를 별도 fragment로 유지한다([[trl-sft-packing-masking]], `data_utils.py` L862-866). 긴 reasoning trace에서는 chunk 경계를 가로지르는 clipping이 packed SFT가 더 나빴던 이유라는 것이 Open R1 저자의 설명이다([[openr1]], Update #3 Lesson 1).
4. **Length-aware loss.** 긴 example과 짧은 example이 pack을 공유하면 per-pack averaging은 sparse pack 안의 example을 선호한다. LongAlign의 \(K/(N_i·M)\) weighting은 example별 동일 weight를 복원한다(§4). ProLong의 SFT는 대신 valid token에 대해 average한다(`--token_scaled_loss`)([[prolong-recipe]], App. A.3).
5. **Memory.** H100 GPU를 쓴 8B ProLong run에서 64K는 sequence parallelism 없이 학습할 수 있는 가장 큰 power-of-2 length였고, 512K stage는 GPU 8개에 걸쳐 DeepSpeed-Ulysses sequence parallelism을 썼다([[prolong-recipe]], App. A.3, A.4). ch-05는 장문 row의 context-parallel layout을, training-memory course(ch-06 kernels, ch-07 parallelism)는 memory mechanism을 다룬다.

## §9 Pre-training 뒤 token을 추가할 때의 tokenizer-template 공동 설계

Pre-training 뒤 `<think>`, tool-call, turn token을 추가하면 model이 한 번도 학습하지 않은 embedding 및 output row가 생긴다.

1. **Initialization.** Transformers의 `resize_token_embeddings`는 기본값으로 `mean_resizing=True`를 쓴다. 이는 기존 embedding의 mean과 covariance를 갖는 multivariate normal distribution에서 새 row를 sample한다. Docstring은 이것이 resizing 전후 next-token distribution의 KL divergence를 줄인다고 말한다([[transformers-chat-templates]], `modeling_utils.py` L2734-2741). Worked example: output embedding이 (1,0), (0,1), (−1,0), (0,−1)이고 hidden state가 (2,1)이면 logit은 2, 1, −2, −1이며 top token probability는 0.696이다. 이 distribution의 중심인 mean (0,0)의 새 row는 logit 0, probability 0.086이며 top token을 0.636으로 낮춘다. 새 row (3,0)는 logit 6, probability 0.974다.
2. **Trainability.** TRL의 `clone_chat_template`은 source tokenizer의 token을 추가하고 dummy token으로 embedding을 64의 배수로 resize한다. PEFT에서는 TRL이 새 id를 trainable token에 추가하고, `modules_to_save`에 `lm_head`가 없으면 “the model may not learn to generate outputs with these new tokens”라고 warn한 뒤 `lm_head`를 `modules_to_save`에 추가한다([[trl-sft-packing-masking]], `chat_template_utils.py` L45-136; `sft_trainer.py` L1088-1108).
3. **Masking.** Model이 emit해야 하는 모든 새 token은 모든 stage에서 loss mask 안에 있어야 한다. §6의 OLMo 3 `<think>`가 그 사례다.
4. **Tokenizer에 template이 없을 때의 default.** Alignment Handbook의 `sft.py`는 `setup_chat_format(..., format="chatml")`을 호출하며, 이 호출은 ChatML token을 추가한다([[hf-alignment-handbook]], `scripts/sft.py` L98-100).

## Negative samples와 negative feedback

이 장의 stage는 course standard의 네 가지 의미 중 두 가지 의미에서 negative를 쓴다. **Content로서의 negative**: 실패한 turn, tool error, rejected draft는 label을 −100으로 둔 채 context에 남기고, corrected continuation만 학습한다(ch-31a). **Gradient로서의 negative**: DPO에서는 rejected response의 log-probability를 아래로 밀며(ch-39, ch-43a), loss mask가 그 push를 받는 token을 결정한다.

**Masking 문제가 생기는 곳.** Chosen/rejected response는 header와 end-of-turn id 같은 formatting token을 공유한다. Llama 3는 두 response 모두에서 이를 mask했다. 저자는 이 token을 loss에 기여하게 두면 tail repetition 또는 termination token을 갑자기 생성할 수 있다고 보고하고, shared token이 같은 token을 올리고 내리는 상충 objective를 만든다고 가정한다([[llama-3]], §4.1.4). 이는 qualitative이며 수치는 보고되지 않았다.

**Mechanism.** Logit \(z\)에 대한 softmax에서,

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

여기서 `y`는 target token, `p_j`는 token j의 probability, `1[·]`는 indicator다. log \(p_y\)를 낮추면 \(z_y\)는 \(1-p_y\)에 비례해 내려가고, 다른 모든 \(z_j\)는 \(p_j\)에 비례해 올라간다. Rejected response의 end-of-turn token이 p = 0.90, newline이 0.08, 다른 모든 token의 합이 0.02라면, 한 번의 push-down step은 logit을 −0.10, +0.08, +0.02에 비례해 바꾼다. 증가분의 80%는 가장 가능성 높은 대안인 newline으로 간다. Response 끝에서 end-of-turn token보다 newline을 선호하도록 학습한 model은 계속 생성하며, 이는 Llama 3가 보고한 tail repetition과 일관된다(**해석**).

**Control.** 두 response에서 shared formatting token을 mask한다. Chosen response에 NLL term을 추가한다(Llama 3는 coefficient 0.2 사용, §4.1.4). Model action이 아닌 tool output과 environment observation은 mask 상태로 둔다(ch-45b).

**진단.** Chosen/rejected log-probability를 각각 기록하고, chosen response 끝에서 end-of-turn token의 probability를 기록하며, length limit에 도달한 response 비율을 추적한다.

**일반성에 미치는 효과.** 모든 response는 끝나야 하므로 termination token의 masking error는 학습 domain뿐 아니라 모든 task에 영향을 줄 수 있다(**해석**; source는 이를 task 전반에서 측정하지 않았다).

## 레시피

| Model (정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama 3.1 (all) | 8B-405B | pretrain-stable | attention mask | 하나의 sequence에서 document 사이 self-attention 없음 | arXiv:2407.21783v3 §3.2 ([[llama-3-recipe]]) | verified 2026-09-14 | standard pre-training에서는 “limited impact”, 매우 긴 sequence에서는 “important”; 수치 없음 |
| Llama 3.1 405B | 405B | SFT | loss masking | target token의 cross-entropy, prompt token은 mask | arXiv:2407.21783v3 §4.1.3 | verified 2026-09-14 | ablation 보고 없음 |
| Llama 3.1 405B | 405B | preference | DPO token masking; NLL term | chosen/rejected에서 header와 termination token mask; chosen의 NLL coefficient 0.2 | arXiv:2407.21783v3 §4.1.4 | verified 2026-09-14 | qualitative: 이 token을 mask하지 않으면 tail repetition 또는 abrupt termination token을 “may lead to”; ablation 수치 없음 |
| DeepSeek-V3 | 671B total / 37B active | pretrain-stable | packing; attention | document packing, cross-sample attention masking 없음 | arXiv:2412.19437v2 §4.1 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| DeepSeek-V3 | 671B total / 37B active | SFT | packing; attention | sequence당 multiple sample, sample masking | arXiv:2412.19437v2 §5.1 | verified 2026-09-14 | ablation 보고 없음 |
| DeepSeek-V4-Flash / -Pro | both | pretrain-stable | packing; attention | truncation을 최소화하도록 document pack, sample-level attention masking | arXiv:2606.19348v1 §4.1 ([[deepseek-v4-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| ProLong-64k-Base | 8B | long-context | attention; packing | cross-document masking; short data를 64K sequence로 pack; long data는 ≥ 64K token document에서 | arXiv:2410.02660v4 §6.1, App. A.2, B.2 ([[prolong-recipe]]) | verified 2026-09-14 | Table 20: mask 사용 시 long / short 54.6 / 65.5, 미사용 시 53.6 / 64.9 |
| ProLong-512k-Instruct | 8B | SFT | loss masking; aggregation | instruction token mask; valid token으로 loss average | arXiv:2410.02660v4 App. A.3; `train_sft.sh` `--apply_instruct_masks`, `--token_scaled_loss` | verified 2026-09-14 | ablation 보고 없음 |
| Llama 2 Long Chat | 70B | SFT | loss masking | output token과 긴 input prompt에 LM loss | arXiv:2309.16039v3 §2.2, §4.3 ([[llama-2-long]]) | verified 2026-09-14 | Table 9: input loss 없음 QuALITY 59.3, 있음 77.3(model size 미명시) |
| Tülu 3 SFT | 8B, 70B | SFT | loss aggregation | sum loss(“opted generally”) | arXiv:2411.15124v5 §4.3.2 ([[tulu-3]]) | verified 2026-09-15 | Fig. 5: Llama 3.0 + Tülu 2 mix에서 sum loss와 LR 5e-6이 최상; 값은 figure에만 있음 |
| Tülu 3 SFT | 8B, 70B | SFT | chat template | Tülu 3 template: model response 전 template 끝에 newline 없음 | arXiv:2411.15124v5 §4.3.1, App. B.3 | verified 2026-09-15 | Table 13: newline을 EOS로 교체하면 53.0, 52.8 대비; 후속 stage의 일관성을 위해 선택하지 않음 |
| Tülu 1 (all runs) | 6.7B-65B | SFT | loss masking | assistant token에만 loss | arXiv:2306.04751v2 §3.2 ([[tulu-1-how-far-can-camels-go]]) | verified 2026-09-14 | ablation 보고 없음 |
| Nemotron-4-340B-Instruct | 340B | SFT | loss masking | user turn mask; assistant turn에 loss | arXiv:2406.11704v2 §3.3.1 ([[nemotron-4-synthetic-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| OpenR1-Distill-7B | 7B | distill-SFT | max length; packing; template | 32,768; `packing: false`; `<think>` section을 요청하는 default system prompt의 ChatML | open-r1@1416fa0 `recipes/OpenR1-Distill-7B/sft/config_distill.yaml` ([[openr1-recipe]]) | verified 2026-09-14 | model card exp7-8: math에서 packing 대 no packing(figure만) |
| OlympicCoder-7B | 7B | distill-SFT | packing | 사용하지 않음 | HF blog “Open R1: Update #3”, Lessons 1 and 4 ([[openr1-recipe]]) | verified 2026-09-14 | 모든 ablated dataset에서 packing이 “consistently worse”(figure만) |
| Alignment Handbook zephyr-7b-beta SFT config | 7B | SFT | template; max length; packing; loss keys | Zephyr `<|user|>`/`<|assistant|>` template와 EOS; `max_seq_length: 2048`; `packing` key 없음; `assistant_only_loss` key 없음 | alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml` L8, L42 ([[hf-alignment-handbook]]) | verified 2026-09-15 | ablation 보고 없음; 2023 release의 config임이 보이지 않음 |
| TRL `SFTConfig` default | any | SFT | packing; strategy; padding-free; loss masks | `packing=False`; `packing_strategy="bfd"`; `padding_free=False`(`bfd` packing이면 이 값과 무관하게 enabled); `completion_only_loss=None`; `assistant_only_loss=False` | trl@aa89588 `trl/trainer/sft_config.py` L77-105 ([[trl-sft-packing-masking]]) | verified 2026-09-15 | framework default; repository에 ablation 없음 |
| LLaMA-2 7B NEFTune runs | 7B | SFT | NEFT α | Alpaca 5, Evol-Instruct 5, ShareGPT 10, OpenPlatypus 15 | arXiv:2310.05914v2 Table 7 ([[neftune]]) | verified 2026-09-14 | AlpacaEval에서 α ∈ {5, 10, 15}의 최상, ChatGPT judge(App. A.1) |
| IM/IT runs (Shi et al.) | 7B, 13B | SFT | LR; batch; epochs; max length | 2e-5; 128; 2, 3, 또는 10; 2,048 | arXiv:2405.14394v2 App. C, Table 6 ([[loss-masking-prompt]]) | verified 2026-09-15 | 이 값의 ablation 보고 없음 |
| PLW study, AlpacaDataShort | 7B | SFT | prompt-loss weight | fitted optimum λ = 0.242 | arXiv:2401.13586v4 §5.2 ([[prompt-loss-weight-sift]]) | verified 2026-09-15 | printed value; Table 3 authors' fitted quadratic의 maximum(AlpacaDataShort에서 p < 0.001) |
| LongAlign ChatGLM3-6B-64k, Llama-2-7B-64k | 6B, 7B | long-context SFT | packing; loss weighting | example별 scale \(K/(N_i·M)\)를 둔 packing | arXiv:2401.18058v1 §3.3, Eq. 4 ([[longalign]]) | verified 2026-09-15 | Table 3: unweighted packing 대비 LongBench-Chat 6.21 vs 5.76, 6.10 vs 5.89 |
| Best-fit packing models (Ding et al.) | 7B, 13B | pretrain-stable | packing; attention; position id | BFD; cross-document attention mask; position id 조정 안 함 | arXiv:2404.10830v2 App. C.1 ([[fewer-truncations-best-fit-packing]]) | verified 2026-09-15 | Table 10의 mask 및 packing ablation |

**작은 범용 run의 시작점.** Mixed instruction set에서 7–8B model을 SFT할 때 Ding et al. pre-training run(7B–13B, BFD, cross-document attention mask, 조정하지 않은 RoPE position id)과 DeepSeek-V3 SFT(sample masking)처럼 BFD와 intra-document attention으로 pack한다. Whole로 학습할 가장 긴 example 이상으로 `max_length`를 설정하거나 continued pre-training data에는 `bfd_split`을 쓴다. TRL 기본 `bfd`가 overflow를 truncate하기 때문이다. Llama 3.1, Tülu 1, Nemotron-4-340B-Instruct처럼 prompt token을 mask한다. Dataset이 short-completion이면 AlpacaDataShort에서 LLaMA 7B에 fitted된 0.242 optimum 주변의 prompt-loss weight를 sweep한다. Long/short example이 pack을 공유하면 64K context에서 6–7B로 시험된 LongAlign의 example별 scaling을 쓴다. Tülu 3 template choice(Table 13)와 Llama 3.1 DPO의 header/termination-token masking은 검증된 template/preference-stage setting이다. 위 row 중 multi-turn data의 template-diversification ratio나 NEFTune α를 검증한 것은 없다.

## 일반화 렌즈

**(a) 무엇이 breadth를 늘리는가.**

- Document를 온전하게 유지하는 것: BFD packing은 13B text model에서 reading comprehension, NLI, context following을 올렸고, 7B code model에서 undefined-name error를 줄였다([[fewer-truncations-best-fit-packing]], Tables 4-5, §4.6).
- Intra-document attention: 1.3B에서 in-context-learning accuracy와 closed-book QA가 높았고([[sequence-composition-intra-doc-masking]], Tables 2-3), 8B, 64K에서 long/short score가 더 좋았다([[prolong]], Table 20).
- Short-completion 또는 작은 data에서 instruction에 loss를 주는 것: memorization이 줄고 held-out score가 높아진다([[loss-masking-prompt]], §4.3; [[prompt-loss-weight-sift]], §5.3).
- Instruction tuning 중 mixed zero-shot/few-shot template: T5-XL에서 zero-shot/few-shot evaluation 모두에서 held-in과 held-out MMLU score가 높았다. 단 zero-shot chain-of-thought와 BBH는 few-shot template 없이 더 높았다([[flan-collection]], §3.2, Fig. 3, Table 1).

**(b) 무엇이 narrowing 또는 forgetting을 일으키는가.**

- Sample이 chunk boundary를 넘게 하는 layout으로 긴 reasoning trace를 pack하는 것: Open R1 code-reasoning SFT ablation에서 LiveCodeBench score가 낮았다. Trace clipping은 저자의 hypothesis이며 측정된 원인은 아니다([[openr1]], Update #3 Lesson 1).
- Single-turn-only data를 pack하는 것: Filtered OpenHermes 2.5에서 MATH 하락이 있었고, 1/40에서 1/20의 multi-turn data가 이를 되돌렸다. 저자의 internal single-turn set에서는 관찰되지 않았다([[packing-analysis-sft]], §5.3).
- Short-completion data에 response-only loss: training completion memorization, PLW ≤ 0.1에서 corpus BLEU 약 80([[prompt-loss-weight-sift]], §5.3.3).
- Template-conditioned behavior: `<think>` prefill이 없으면 out-of-domain prompt가 base behavior로 되돌아갔다([[openr1]], Update #3 Lesson 4).
- Instruction loss의 대체로 base model KL regularization을 쓰는 것: KL 없는 IT 대비 LIMA AlpacaEval 2.0은 2.58에서 0.06으로, Alpagasus Dolly 9k에서는 2.28에서 0.04로 내려갔고, NLP mean은 48.79에서 49.26, 45.54에서 49.31로 올랐다(LLaMA-2-7B; [[loss-masking-prompt]], Table 3).

**(c) 이 stage에서 어떻게 측정하는가.**

- Multiple-choice와 generation benchmark를 분리해 보고한다. NEFTune과 PLW는 judge-scored long generation을 움직였지만 multiple-choice score는 안정적이거나 반대 방향으로 움직였다([[neftune]], Fig. 3; [[prompt-loss-weight-sift]], §5.1).
- 각 checkpoint를 여러 형식에서 평가하고 spread와 ranking change를 보고한다([[prompt-format-sensitivity-formatspread]], §3.2; [[mind-your-format-templates]], §5).
- Template 변경을 수용하기 전에 template effect를 seed variance와 비교한다([[tulu-3]], Tables 13-14).
- Training prompt에서 greedy output의 BLEU로 training completion memorization을, 별도 instruction set의 held-out loss를 측정한다([[loss-masking-prompt]], §4.3).
- Long-context 변경은 SFT 뒤 downstream long task와 short task를 함께 평가한다([[prolong]], §2.1-2.3).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| Isolation이 의도됐는데 intra-document attention 없이 packing | 오류 없음; §2 pre-training study에서 더 높은 perplexity와 더 낮은 in-context-learning score | `cu_seqlens` segment 수가 example 수와 같은지 assert; pack된 example과 단독 example의 logit 비교 |
| Padding-free path에서 all-ones `attention_mask` 전달 | FlashAttention이 `position_ids`를 무시하고 example 사이 attention 수행 | `padding_free=True`일 때 `attention_mask` key가 없는지 assert(TRL `sft_trainer.py` L485-486) |
| Boundary를 `position_ids`에서 추론하는 framework에서 continuous `position_ids` | Row당 inferred segment 하나 | `position_ids`에서 boundary를 재계산; example당 start 하나를 기대 |
| Example 첫 token을 이전 example 마지막 token의 target으로 학습 | Example을 가로지르는 loss term | `labels[position_ids == 0] == -100` assert |
| [B, T] tensor에서 `labels[:prompt_len] = -100` | 처음 `prompt_len`개 example은 trained token이 0, 나머지는 prompt token으로 학습 | Row별 trained-token fraction 기록 |
| `{% generation %}` 없는 template에 `assistant_only_loss=True` | Patch할 수 없는 template에서 TRL `ValueError`; patched family에서는 tokenizer template과 다르게 conversation을 render할 수 있음(§6 Qwen3 tool-call history) | Example별 trained token 수 count; masked span decode; training template과 serving template diff |
| Assistant end-of-turn token이 mask 밖 | Model이 멈추지 않고 output이 length limit 도달 | 각 assistant turn의 마지막 trained token decode; TRL `is_chat_template_stop_token_trained` |
| SFT, DPO, RL, serving 간 template 상이 | Output formatting 변경; stop token 누락 | 각 tokenizer로 conversation 하나를 render하여 id diff(open-instruct `diff_tokenizers.py`, `docs/olmo3.md` L47-51) |
| `tokenize=False` rendering 뒤 BOS 중복 | Sequence start에 BOS id 두 개 | Rendered example마다 첫 두 id 검사 |
| Training data와 release의 generation prompt 상이 | Behavior가 in-domain prompt에만 나타남 | Release template로 out-of-domain prompt 평가 |
| Wrapped packing 또는 긴 example의 `bfd` truncation | Answer가 question과 분리; trace truncate | Token이 drop/split된 example 수 count |
| Mixed length와 gradient accumulation에서 token-mean aggregation | Accumulation step 또는 GPU 수에 따라 결과 변화 | Example별 loss weight 기록; sequence mean과 비교(Tülu 3 Eqs. 1-2) |
| Packed row에서 NEFTune | Noise scale이 paper의 per-sequence 값보다 낮음 | Row length 및 example length로 α/√(L·d) 계산 |
| Single-format evaluation | 다른 형식에서 ranking 변경 | 동등 형식 sample을 평가(FormatSpread는 task당 10개, §4.2)하고 spread 보고 |
| PEFT에서 frozen `lm_head`와 새 token | 새 token을 생성하지 않음 | 새 token이 target인 위치에서 probability 점검 |

## 이해를 점검하세요

1. 올바른 block-diagonal mask와 RoPE가 있을 때 position id reset이 왜 attention을 바꾸지 않는지, 그리고 Hugging Face padding-free path에서는 reset이 왜 여전히 중요한지 설명하라.
2. f(i)를 써서 document masking이 긴 relative distance 학습을 왜 줄일 수 있는지, long-context stage에서 어떤 data property가 그 감소를 피하는지 설명하라.
3. Fewer Truncations ablation은 concatenation+mask가 perplexity를 개선하지만 summarization을 낮춤을 보인다. BFD packing은 mask 외에 무엇을 바꾸며, 그것이 추가 이득을 설명할 수 있는가?
4. Completion이 짧을수록 instruction loss가 overfitting을 더 줄이는 이유와, base model에 KL penalty를 주는 것이 왜 같은 benefit을 주지 못할 수 있는지 설명하라.
5. Per-turn unrolling과 all-assistant masking이 동일한 per-token loss를 만든다는 것을 보이고, 이 논증에 필요한 template property를 이름 붙여라.
6. LongAlign weighting worked example에서 per-pack token averaging은 어떤 example을 선호하며, \(K/(N_i·M)\) scale은 그 선호를 어떻게 제거하는가?
7. Llama 3는 DPO response 둘에서 formatting token을 mask했다. Softmax gradient를 이용하여 shared end-of-turn token을 아래로 미는 일이 종료하지 않는 response를 어떻게 만들 수 있는지 설명하라.
8. 새 template이 SFT model average score를 1.4 points 올렸다. 이 이득을 seed variance나 evaluation-format choice가 아니라 template 탓으로 돌리기 전에 어떤 추가 측정이 필요한가?

## 연결

- 의존: ch-00 — What General Capability Means and How It Is Measured.
- 순서상 이전: ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization.
- 다음: ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen(긴 row의 step당 token과 context parallelism).
- ch-07 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures(masking failure를 진단 category로 다룸).
- ch-11 — Tokenizers, Data Provenance, and PII Removal(special token 추가 전 tokenizer 구성).
- ch-26 — Tool and Function-Calling Data(tool-call 및 tool-response format).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks(instruction collection의 template diversity).
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate(§4–§7을 SFT regime에 적용).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood(masked failure를 content로 사용).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation(§3, §8).
- ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split(mistake table의 test).
- ch-39 — Offline Preference Optimization: DPO and Its Variants; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation(format spread를 perturbation audit로 사용).

## 출처

- [[sequence-packing]] — Krell et al. 2021(BERT): padding share, SPFHP/NNLSHP efficiency, mask/position-id ablation, per-pack loss weighting. 2026-07 날짜의 library card version은 SPFHP를 잘못 설명하고 equivalence를 과장하므로 excerpt를 사용한다.
- [[fewer-truncations-best-fit-packing]] — Ding et al. 2024: BFD algorithm, compactness, capability/hallucination result, cross-document mask ablation, RoPE position id 미조정.
- [[sequence-composition-intra-doc-masking]] — Zhao et al. 2024: 1.3B에서 perplexity, in-context learning, QA에 대한 intra-document causal masking 결과.
- [[packing-flash-attention-position-ids]] — Kundu et al. 2024: `position_ids`에서 `cu_seq_len`, first-label masking, packing 대 padding validation loss.
- [[packing-analysis-sft]] — Wang et al. 2024: 8B/70B SFT의 packing 대 padding, batch-size/LR 관계, single-turn packing MATH 하락.
- [[loss-masking-prompt]] — Shi et al. 2024: instruction modelling result, overfitting 측정, KL 비교, training setting. 2026-07 날짜의 library card version은 paper finding을 뒤집으므로 excerpt를 사용한다.
- [[prompt-loss-weight-sift]] — Huerta-Enochian & Ko 2024: prompt-loss weight regression, λ = 0.242, mechanism analysis.
- [[neftune]] — Algorithm 1, AlpacaEval result, overfitting/length control, dataset별 α.
- [[prompt-format-sensitivity-formatspread]] — Sclar et al. 2023: format spread 정의와 측정.
- [[mind-your-format-templates]] — Voronov et al. 2024: 21개 model 전반의 template sensitivity와 낮은 template transfer.
- [[flan-collection]] — Longpre et al. 2023: mixed zero-shot/few-shot template ablation.
- [[hf-alignment-handbook]] — 1de1fc9의 Zephyr SFT config와 `sft.py`. Library card의 `train_on_response_only` snippet은 TRL에 없으므로 excerpt를 사용한다.
- [[trl-sft-packing-masking]] — aa89588의 TRL: packing strategy, padding-free collator, assistant mask check, template cloning, prefix-preservation check.
- [[transformers-chat-templates]] — 05e078a의 Transformers: chat template docs, NEFTune hook, `mean_resizing`.
- [[open-instruct-sft-masking]] — d8a7f1c의 open-instruct: all-assistant 및 last-turn masking function.
- [[open-instruct-allenai-recipes]] — `docs/olmo3.md`: stage 간 OLMo 3 template 차이와 `<think>` masking bug.
- [[llama-3]], [[llama-3-recipe]] — document mask, chat dialog format, SFT prompt masking, DPO formatting-token masking.
- [[prolong]], [[prolong-recipe]] — document-masking ablation, variable-length throughput, long-data layout, SFT masking/aggregation.
- [[longalign]] — packing 아래 example별 loss weighting과 LongBench-Chat 효과.
- [[string-effective-context]] — relative-position frequency f(i)와 effective context length의 관계.
- [[tulu-3]] — batch-aggregation equation, sum loss, template variation(Table 13), seed variance(Table 14).
- [[llama-2-long]] — 긴 input prompt의 LM loss(Table 9).
- [[openr1]], [[openr1-recipe]] — packing lesson, `<think>` prefill lesson, OpenR1-Distill-7B config.
- [[deepseek-v3-recipe]], [[deepseek-v4-recipe]] — version 전반의 packing/sample-masking practice.
- [[tulu-1-how-far-can-camels-go]], [[nemotron-4-synthetic-recipe]] — 배포 recipe에서 assistant-only loss masking.
