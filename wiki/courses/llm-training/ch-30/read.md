<!-- chapter: ch-30
     track: sft
     kind: content
     title: SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate
     deps: [ch-29e, ch-04]
     sources: [[loss-masking-prompt]], [[loss-masking-regimes]], [[neftune]], [[sequence-packing]], [[sequence-packing-contract]],
              [[hf-alignment-handbook]], [[tulu-3]], [[tulu-3-sft-recipe]], [[open-instruct-allenai-recipes]], [[chat-template-matrix]],
              [[smol-training-playbook]], [[lora-learns-less-forgets-less]], [[lora-without-regret]], [[minimax-m2-interleaved-thinking]],
              [[glm-5]], [[orpo]], [[qwen-3]], [[llama-3]], [[llama-3-recipe]], [[qwen-2.5-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[in2-film]]
     figures: figures/sft-axes.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate

> **Core insight.** Supervised fine-tuning (SFT) choices that look like implementation details change how much a model overfits its instruction data. On single-turn instruction sets of about 1K to 70K examples, adding loss on instruction tokens raised AlpacaEval 1.0 on 6 of 7 datasets and adding embedding noise raised AlpacaEval on 4 of 4 datasets; in the datasets where each group measured it, both methods gave higher training loss, lower held-out loss, and lower overlap with training responses ([[loss-masking-prompt]] Table 1, Fig. 3, Table 2; [[neftune]] Table 1, Figs. 4-5). Trainer settings also change results: Tülu 3 chose a sum loss after tracing a gap between training setups to loss aggregation, and its SFT runs with different seeds span 0.3 average points at 8B and 2.6 at 70B ([[tulu-3]] §4.3.2, Table 14).
>
> **Guideline.** When the SFT set is small (about 1K-14K examples) or its instructions are long relative to outputs, evaluate loss over instructions against response-only loss, because Shi et al. measured the largest gains in that regime. Otherwise keep assistant-only loss on all assistant turns, because that is the open-instruct default and SmolLM3 reports that masking user turns improved most of its evaluations. When copying a published learning rate, also copy its loss aggregation (sum or mean), because Tülu 3 selected 5×10⁻⁶ with a sum loss. When packing a small dataset, compare against an unpacked run at the same number of optimizer steps, because packing reduces steps per epoch. Choose every setting on a development suite and report on a held-out suite that was not examined during development.

## Corrections to the version you studied

1. "RL moves the policy *along* [what SFT shaped]; it does not rebuild it" and "If SFT never emitted a tool-call, PPO cannot find one" → DeepSeek-R1-Zero applied GRPO (a policy-gradient RL algorithm) directly to DeepSeek-V3-Base with no SFT and raised AIME 2024 pass@1 from 15.6% to 77.9%; what SFT and RL each add is measured in later chapters, not settled by SFT ([[deepseek-r1]], v2 §2, §2.3).
2. "response-only SFT strictly beats full-sequence SFT on MT-Bench and AlpacaEval" and "Response-only dominates … full-sequence loses broadly" → Shi et al. report that loss over instructions improves results "in many scenarios", with the largest gains for long instructions with short outputs and for few examples; on Less MMLU Chat, AlpacaEval 1.0 rose from 4.42 to 9.78 ([[loss-masking-prompt]], arXiv:2405.14394v2 Abstract, Table 1).
3. "Shi 2024 … ablated three variants — response-only, full-sequence, prompt-weighted" → the prompt-loss-weight study is Huerta-Enochian and Ko (arXiv:2401.13586); Shi et al. compare instruction modelling with instruction tuning and NEFTune ([[loss-masking-regimes]]; arXiv:2405.14394v2 §4.1).
4. "mask every `u_i`, mask `a_1..a_{k-1}`, train only on `a_k`" as the default → open-instruct's default SFT transform trains every assistant turn (`last_turn_only=False`), and TRL's `assistant_only_loss` computes loss on all assistant responses; last-turn-only is a separate option ([[loss-masking-regimes]]: open-instruct@098424c `dataset_transformation.py` L1214-1218, L1554-1575; TRL v0.23.0 `sft_config.py` L85-94).
5. "unrolls the conversation k times … same loss value, k× more data" → per-turn expansion adds no new target tokens; with identical history rendering it yields the same (context, target) pairs as single-pass loss on all assistant turns and re-processes each prefix. It differs only when the template renders earlier turns differently from the current turn (derivation in §1.2; rendering rule in [[chat-template-matrix]], Qwen3-8B model card, Best Practices).
6. "| Template | ChatML (Mistral default) |" and "ChatML (OpenAI / early Mistral)" → the Zephyr-7B-β SFT config renders `<|system|>`, `<|user|>`, `<|assistant|>` followed by content and `eos_token`; FILM-7B, fine-tuned from Mistral-7B-Instruct-v0.2, used the Mistral `[INST]` template ([[hf-alignment-handbook]], alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml`; [[in2-film]] §2.2, App. D Example 9).
7. Zephyr table rows "Packing true", "Train on response only true", "NEFTune α=5", "AdamW (β₁=0.9, β₂=0.95)", "FSDP FULL_SHARD" → the config sets none of these keys; it sets `learning_rate: 2.0e-05`, cosine schedule, `warmup_ratio: 0.1`, 1 epoch, `max_seq_length: 2048`, per-device batch 16, gradient accumulation 1. The Trainer-generated card of the Hub model `alignment-handbook/zephyr-7b-sft-full` reports Adam with betas (0.9, 0.999) and a total batch of 128 over 8 devices, so the old "Global batch 128" row agrees with the model card and the β₂ = 0.95 row does not ([[hf-alignment-handbook]], same file; huggingface.co/alignment-handbook/zephyr-7b-sft-full README).
8. "`SFTConfig(packing=True, train_on_response_only=True)`" → TRL's `SFTConfig` has `completion_only_loss` (prompt-completion datasets) and `assistant_only_loss` (conversational datasets, default `False`); it has no `train_on_response_only` field ([[loss-masking-regimes]], TRL v0.23.0 `sft_config.py` L85-94).
9. Tülu 3 "| Template | Llama-3 (native) |" → Tülu 3 used its own template; in Table 13 the Llama 3 template scored lowest, 51.6 against 52.8 for the Tülu 3 template ([[tulu-3]], arXiv:2411.15124v5 §4.3.1, Table 13, App. B.3).
10. "[[allenai-tulu-sft-recipe]] finds NEFTune neutral at 939K SFT prompts" and the rule "on when |D| ≤ 100K, off at |D| ≥ 500K" → the Tülu 3 report does not mention NEFTune, and no source gives that size rule ([[tulu-3]] §4.3; [[neftune]] Verification).
11. Tülu 3 table rows "Packing yes", "AdamW (0.9, 0.95)", "FSDP FULL_SHARD / HYBRID_SHARD" → none appear in Table 11 or §4.3; the report gives LR, linear schedule, effective batch 128, maximum length 4,096, warmup ratio 0.03, 2 epochs, and a sum loss, which the old table omitted; the reproduction command uses DeepSpeed ZeRO-3 ([[tulu-3]] Table 11, §4.3.2; [[open-instruct-allenai-recipes]] Technical Details).
12. "70B parameter count halves the learning rate" → 2×10⁻⁶ against 5×10⁻⁶ is a factor of 2.5, and each value was chosen by a hyperparameter search ([[tulu-3]] §4.3).
13. "Raw instruction batches are 50–89% padding" → Krell et al. measured BERT pre-training data (Wikipedia, about 50% padding) and GLUE (89% for CoLA at length 128); no instruction data were measured ([[sequence-packing]], arXiv:2107.02027 Abstract).
14. "speedup ≈ L_max / avg(L_i) … expected 6×, realised 2.5–3×" and "loss curves must agree within 0.01 nats" → neither appears in any source; Kundu et al. found higher validation loss for offline packing with correct position IDs (1.284 against 1.129 for padding) and attribute it to far fewer optimizer steps in one epoch ([[sequence-packing-contract]], arXiv:2407.09105 §4.1, Table 2).
15. "Four failure modes, each enumerated in packed-vs-unpacked-ablation" and "Un-reset position IDs → RoPE is position-shifted" → that card has no primary source; with a correct block-diagonal mask, rotary attention depends only on relative offsets, so a constant offset inside one document does not change it. Position IDs matter when the kernel derives boundaries from them (Kundu §3.3) or when positions are learned absolute embeddings (BERT: MLM accuracy stalls at 71.8% against 72.1%) ([[sequence-packing-contract]]; arXiv:2107.02027 §4.2.1).
16. "downstream win-rate collapses 5-20 pts" and "[[hf-alignment-handbook]] names it the '#1 silent bug'" → no source gives 5-20 points, and the statement was not found in the handbook files checked (`config_full.yaml`, `scripts/sft.py` at 1de1fc9); the measured template effect in Tülu 3 is a 1.4-point spread across five templates, which measures template choice, not a mismatch between training and serving ([[tulu-3]] Table 13).
17. "that the base model's post-training already conditioned on" → a base checkpoint has no post-training; Llama 3 describes its header and termination tokens as a chat protocol designed for post-training ([[llama-3]], arXiv:2407.21783v3 §4.1.1).
18. NEFTune "+8–10 pts" on Evol-Instruct, ShareGPT, OpenPlatypus → +9.26, +7.54, +8.61 AlpacaEval points with a GPT-4 judge ([[neftune]] Table 1).
19. "SFT … leaves every other behaviour on the base model's prior" → the softmax cross-entropy gradient lowers every non-target logit in proportion to its probability, and SFT on chosen responses raised the log-probability of rejected responses in the ORPO experiment ([[orpo]] §3, Fig. 3).
20. "ch-32 (reasoning SFT and long-CoT cold-start)", "ch-33 (tool-call SFT)", "ch-34 (agentic SFT)", "populate the mix in ch-31..ch-35" → ch-32 is mid-training, ch-33 and ch-34 are case studies, tool and agentic data are ch-26 and ch-27, and SFT mixture construction is ch-30b (outline.json).

## Why this chapter matters for a general-purpose model

The pipeline order is pre-training → mid-training → SFT → preference optimization → RL → evaluation. SFT trains on curated demonstrations rendered in a chat format. It sets the output format that preference optimization and RL later sample from, and training on a small set can lower scores the base model had on tasks outside that set (first problem below).

Three measurable problems motivate the chapter. First, instruction-tuned models can overfit their SFT data: Shi et al. report that LLaMA-2-7B trained on 9,229 Dolly examples fell from 49.32 to 45.54 on the mean of 18 NLP tasks ([[loss-masking-prompt]] Table 3). Second, trainer details change results: Tülu 3 found a gap between SFT models trained with open-instruct and models trained in other settings, and traced it to loss aggregation ([[tulu-3]] §4.3.2). Third, a choice can raise a development score without raising the matching held-out score ([[tulu-3]] Table 32; §9 below).

[[ch-29e]] covered which data properties give unseen-task generalization, and [[ch-04]] covered the mechanics of packing, masks, and templates. This chapter asks which settings of those mechanics generalize, and how to measure it.

## §1 SFT regimes and the loss-mask rule

### 1.1 Which tokens receive loss

**Definition.** A loss mask is a per-token label vector in which positions set to `-100` are ignored by cross-entropy. An SFT regime is the shape of one example: single-turn, multi-turn, tool call, reasoning (with a thinking span), or agentic (a multi-step trajectory with tool calls and tool outputs).

**Problem.** Tokens that the model never produces at inference (user text, tool outputs) can be trained as targets, and tokens that it must produce (earlier assistant turns, tool-call arguments) can be excluded. Either change alters what the model learns without raising an error.

| Segment | Produced at inference by | Default label in open-instruct and TRL assistant-only loss | Variant with evidence |
|---|---|---|---|
| System prompt, tool schemas | client | masked | not tested with loss in the chapter sources |
| User turns | user | masked | single-turn instruction tokens included (minus template tokens) under instruction modelling, [[loss-masking-prompt]] Eq. 4 |
| Assistant turns, all of them | model | trained | last turn only: open-instruct `last_turn_tulu_tokenize_and_truncate_v1` |
| Tool-call text inside an assistant message | model | trained (it is assistant content) | — |
| Tool outputs (`tool` role) | environment | masked (only `role == "assistant"` spans are trainable) | — |
| Thinking span of an assistant turn | model | trained if rendered | earlier-turn thinking may not be rendered at all (§7) |
| Erroneous steps in an agentic trajectory | model | trained unless removed | GLM-5 keeps them in context and masks them from the loss ([[glm-5]] §3.1) |

The open-instruct code that selects trainable spans ([[loss-masking-regimes]], open-instruct@098424c `open_instruct/dataset_transformation.py` L1214-1218):

```python
def _trainable_assistant_indices(messages: list[dict[str, Any]], last_turn_only: bool) -> list[int]:
    assistant_indices = [idx for idx, m in enumerate(messages) if m["role"] == "assistant"]
    if last_turn_only:
        return assistant_indices[-1:]
    return assistant_indices
```

The default SFT transform is `sft_tulu_tokenize_and_truncate_v1` (`finetune.py` L132-133), which calls this with `last_turn_only=False`. TRL v0.23.0 documents the defaults: `assistant_only_loss` "defaults to `False`", in which case "loss is computed on the entire sequence", and `completion_only_loss=None` computes loss on the completion only for prompt-completion datasets (`trl/trainer/sft_config.py` L85-94). A conversational dataset stored as a single `messages` list and trained with TRL defaults therefore trains on user turns.

### 1.2 Single pass over all turns versus per-turn expansion

**Mechanism.**
1. Single pass: render the whole conversation once and train every assistant span. Assistant turn i is predicted from the rendered history before it.
2. Last turn only: same rendering, labels only on the final assistant turn.
3. Per-turn expansion: create one example per assistant turn i, containing the history rendered as inference would render it at turn i, with labels only on turn i.

If the template renders history the same way at every turn, options 1 and 3 produce the same set of (context, target) pairs, so their summed gradients are equal. They differ when the history rendering changes between turns, as in the Qwen3 template, which removes earlier thinking content from history ([[chat-template-matrix]], Qwen3-8B model card, Best Practices).

**Worked example.** A conversation has u₁ = 50 tokens, a₁ = 200 thinking + 50 answer tokens, u₂ = 40 tokens, a₂ = 300 thinking + 60 answer tokens (template tokens ignored).
- Single pass with thinking kept in history: 50 + 250 + 40 + 360 = 700 tokens processed; 250 + 360 = 610 tokens trained. Turn 2 is conditioned on a 340-token history.
- Last turn only: 700 tokens processed; 360 trained.
- An inference client that strips earlier thinking shows turn 2 a history of 50 + 50 + 40 = 140 tokens, which the single-pass example never contains. Per-turn expansion matching that client: copy 1 is 300 tokens (250 trained), copy 2 is 140 + 360 = 500 tokens (360 trained). Total 800 tokens processed, 610 trained, 14% more compute than the single pass.
- With a loss averaged within each sequence and then over the two sequences, a token in copy 1 has weight 1/(2·250) = 0.0020 and a token in copy 2 has weight 1/(2·360) ≈ 0.0014, while the single pass gives every trained token 1/610 ≈ 0.0016; with a sum loss every trained token has weight 1 in both layouts (§3).
- A template that removes earlier thinking during rendering, such as open-instruct's `olmo_thinker_remove_intermediate_thinking` (`content.split('</think>')[-1]` for non-final assistant messages), used in a single pass removes a₁'s 200 thinking tokens from both input and labels: 500 tokens processed, 410 trained ([[chat-template-matrix]], open-instruct@098424c `dataset_transformation.py` L427-471).

The companion figure [figures/sft-axes.html](figures/sft-axes.html) renders this conversation token by token; it lets the reader set the loss layout, the history rule of the training template, and the history rule of the serving client separately, and shows which tokens receive labels, each token's loss weight, and where training and serving histories differ.

**Implication for a general-purpose model.** Training all assistant turns uses every demonstration token. When the deployed client renders history the same way as the training template, a single pass is sufficient. When the client removes earlier thinking, per-turn expansion is the only one of the three layouts that trains every turn, including its thinking, under the history the model receives at inference (Interpretation derived from the rendering rules above).

## §2 Loss over instructions as a regularizer

**Definition.** Instruction tuning (IT) in Shi et al. is the loss over completion tokens only. Instruction modelling (IM) adds loss on instruction tokens and excludes template tokens ([[loss-masking-prompt]] §3):

```
IT:  L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})                         (Eq. 2)
IM:  L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)                    (Eq. 4)
```

Here I is the instruction of m tokens, C the completion of n tokens, x the concatenated sequence, and T the set of template tokens such as `<|user|>`; 1(·) is 1 when the token is not a template token.

**Problem.** Shi et al. hypothesize that training only on the outputs of a small dataset, especially short outputs, lets the model memorize those outputs (§1). They measure memorization with training loss, held-out loss, and BLEU (an n-gram overlap score, 0-100) between greedy outputs for training prompts and the training outputs.

**Evidence** (LLaMA-2-7B base; LR 2×10⁻⁵, batch 128, typically 2 epochs; App. C Table 6). NLP mean is the average of 18 benchmark tasks; AlpacaEval 1.0 is the win rate against Text-Davinci-003 judged by GPT-4; MT-Bench is a GPT-4-judged score (1-10) on 80 two-turn questions. Result (single study):

| Dataset (examples; instruction/output length ratio) | NLP mean of 18 tasks, IT → IM | AlpacaEval 1.0, IT → IM | MT-Bench, IT → IM |
|---|---|---|---|
| Less MMLU Chat (13,533; 26.3) | 47.18 → 47.84 | 4.42 → 9.78 | 3.86 → 4.54 |
| Less Tydiqa (13,533; 5.9) | 48.21 → 48.70 | 5.12 → 10.10 | 4.08 → 4.36 |
| Alpagasus Dolly 9k (9,229; 0.30) | 45.54 → 48.00 | 21.54 → 30.77 | 4.33 → 4.55 |
| Alpagasus Alpaca 5k (5,305; 0.57) | 45.29 → 47.47 | 16.29 → 19.52 | 3.62 → 3.48 |
| LIMA (1,030; 0.09) | 48.79 → 49.60 | 33.06 → 32.94 | 4.77 → 4.83 |

Sources: Table 1 and Table 5 of arXiv:2405.14394v2. IM lowered MT-Bench on Alpaca 5k and Dolly 3k and AlpacaEval 1.0 on LIMA, so the effect is not uniform. The mechanism evidence: on LIMA, mean training loss on output tokens was 1.37 for IT and 1.45 for IM, while test loss on a 10% sample of Tülu V2 was 1.32 for IT and 1.17 for IM (Fig. 3); BLEU between greedy outputs and training outputs fell for IM on all seven datasets, for example 60.96 → 53.94 on Less BBH ICL (Table 2). A KL penalty to the base model is not a substitute: on LIMA it cut AlpacaEval 2.0 from 2.58 to 0.06 (Table 3). The authors state that IM is not proposed as a replacement for current practice (Abstract). The AlpacaEval 1.0 gain on Tülu V2 subsets grew as the number of examples fell from 35,000 toward 1,000 at a fixed instruction/output ratio near 10 (Fig. 2 right, App. C).

**Prompt loss weight.** Huerta-Enochian and Ko define PLW as a weight between 0 (masked prompt) and 1 (full loss) on prompt tokens. With LLaMA-1 7B and LLaMA-2 7B, 10 PLW values, and 3 Alpaca variants (60 runs, 13 benchmarks), performance on short-completion data had a statistically significant negative quadratic relationship with PLW: values 0.01-0.5 were best on multiple-choice and short-generation benchmarks, and values near 1.0 were best on long-generation benchmarks; PLW "can be safely ignored" for long-completion data ([[loss-masking-regimes]], arXiv:2401.13586v4 Abstract, §1 contributions, §4 setup). Result (single study).

**Worked example.** A prompt of 200 tokens has mean token loss 2.0 (sum 400); its completion of 20 tokens has mean loss 1.5 (sum 30). The weighted sum is PLW·400 + 30. At PLW = 0 the loss is 30. At PLW = 0.1 it is 70, and the prompt contributes 40/70 = 57% of it. At PLW = 1 it is 430, and the prompt contributes 93%. With short completions, even a small weight makes the prompt terms the larger part of the summed loss.

**Conditions and limits.** Both studies use models of 6.7B to 13B parameters (LLaMA family; OPT-6.7B in Shi et al. Fig. 5) and LLM-judged win rates. The Shi et al. Table 1 datasets and the Huerta-Enochian and Ko datasets are single-turn; Shi et al. Fig. 2 also includes ShareGPT (50,000 examples) and Tülu V2 (326,181), which contain multi-turn conversations, and the paper does not describe how IM treats earlier turns. SmolLM3 (3B, hybrid reasoning SFT mixture) reports that masking user turns gave "a few points of improvement" in most evaluations, with the largest effect on IFEval ([[smol-training-playbook]], "Masking User Turns"); the text gives no per-benchmark numbers. No study in the library tests IM on a million-example multi-turn mix. **Implication (Interpretation):** the evidence supports loss over instructions as a regularizer for small single-turn sets, especially with long instructions and short outputs; it does not support it as a default for large multi-turn chat mixes.

## §3 Epochs, learning rate, and loss aggregation

**Definition.** Loss aggregation is how per-token losses are combined into the scalar that is differentiated: a mean over non-padding tokens in a forward pass, a mean per micro-batch averaged across accumulation steps, or a sum.

**Problem.** Tülu 3 found a performance gap between SFT models trained with open-instruct and models trained in other settings such as TPUs, and attributed the gap to averaging "without taking into account gradient accumulation or distributed training setups" ([[tulu-3]] §4.3.2).

**Formula** ([[tulu-3]] §4.3.2, Eqs. 1-2). For two samples with summed token losses l₁, l₂ and n₁, n₂ non-padding tokens:

```
one forward pass:        L = (l₁ + l₂) / (n₁ + n₂)          (every token weighted equally)
gradient accumulation:   L = (l₁/n₁ + l₂/n₂) / 2            (every sample weighted equally)
sum loss:                L = l₁ + l₂                          (every token weight 1; LR must be re-tuned)
```

**Worked example.** Sample 1 has n₁ = 10 tokens with mean loss 2.0 (l₁ = 20); sample 2 has n₂ = 90 tokens with mean loss 1.0 (l₂ = 90). One forward pass gives L = 110/100 = 1.10, and each token has weight 0.01. Accumulation over two micro-batches gives L = (2.0 + 1.0)/2 = 1.50; a token of sample 1 has weight 1/20 = 0.05 and a token of sample 2 has weight 1/180 ≈ 0.0056, a 9:1 ratio. The sum gives L = 110 with weight 1 per token, so the gradient scale grows with the number of tokens per batch. Changing the number of GPUs or accumulation steps changes the first two losses but not the third. The aggregation panel of [figures/sft-axes.html](figures/sft-axes.html) recomputes these weights for any n₁, n₂, and micro-batch split.

**Evidence.** Fine-tuning Llama 3.0 on the Tülu 2 mixture, a sum loss with LR 5×10⁻⁶ gave the best average, and 2 epochs beat longer training (Figs. 5-6); Tülu 3 then used sum loss, 5×10⁻⁶ (8B) or 2×10⁻⁶ (70B), and 2 epochs ([[tulu-3]] §4.3). The released open-instruct commands passed `--reduce_loss sum` at commit 8781471, while at 098424c the flag is gone and the default is mean ([[open-instruct-allenai-recipes]] Technical Details). Result (single study) with a documented reproduction conflict.

**Epochs and learning rate, other evidence.** LLaMA-2 7B trained on Alpaca (AlpacaEval win rate with a ChatGPT judge) scored 40.50, 48.26, 48.94, 48.63, and 47.45 without NEFTune and 55.09, 62.55, 62.24, 60.50, and 58.14 with NEFTune at 1, 3, 5, 7, and 9 epochs ([[neftune]] Table 14). Shi et al. plot the 18-task NLP mean over epochs 2-10 on five datasets and report that IM "generally has a lower instruction tuning tax" than IT (Fig. 4); the instruction tuning tax is the drop on these NLP tasks relative to the base model. SmolLM3's LR scan found 3×10⁻⁶ or 1×10⁻⁵ better on average than larger values, with AIME25 falling at LRs above 1×10⁻⁵; 5 epochs gave "a few more percentage points" on average than 1 epoch ([[smol-training-playbook]], "Tuning the Learning Rate", "Scaling the Number of Epochs"). The best epoch count differed by setting: 5 without and 3 with NEFTune (of 1, 3, 5, 7, 9) on Alpaca, 2 for Tülu 3 on the Tülu 2 mix (of 2 to 7), and 5 better than 1 for the SmolLM3 baseline mixture (other counts appear only in the playbook's figure). Each result is a single study; a general rule is an open question.

**Seed variance and soups.** A model soup is the element-wise average of the weights of several models fine-tuned from the same base. Tülu 3 SFT averages by seed were 59.9, 60.1, 59.8, 59.8, 59.8 at 8B (spread 0.3) and 71.8, 70.0, 72.6 at 70B (spread 2.6); the best soup of two seeds scored 60.2 and 72.5, and the best single run was released ([[tulu-3]] Table 14). A difference between two SFT configurations that is smaller than the seed spread at that size is not evidence. Soups are covered in [[ch-30c]].

**Implication.** A learning rate is defined jointly with its aggregation, batch size, and epoch count. When a published learning rate is reused, reuse the other three settings as well, because Tülu 3 selected its LR together with a sum loss and 2 epochs (Figs. 5-6). Otherwise sweep the learning rate again.

## §4 Packing and its effect on optimization

**Definition.** Packing concatenates several examples into one row of fixed length. With variable-length attention (`cu_seqlens`), tokens attend only within their own example; mechanics are in [[ch-04]].

**Problem.** Packing changes the number of optimizer steps and the tokens per step, not only speed.

**Worked example.** 100,000 examples average 500 tokens; batch 128. Unpacked: ⌈100,000/128⌉ = 782 steps per epoch, about 128 × 500 = 64,000 non-padding tokens per step on average. Packed into 4,096-token rows with full fill: 50,000,000/4,096 = 12,208 rows, ⌈12,208/128⌉ = 96 steps per epoch, 524,288 tokens per step. The same epoch count now gives 96 instead of 782 optimizer updates, a factor of 8.1.

**Evidence.**
- Kundu et al. (Mistral-7B, FLAN 20K subset, one epoch): padding 742 tokens/s and validation loss 1.129; offline packing with position IDs into 585 rows 3,010 tokens/s and 1.284; mini-batch packing that keeps the step count 1,408 tokens/s and 1.127 ([[sequence-packing-contract]], arXiv:2407.09105 §4.1, Table 2).
- Wang et al. (LLaMA-3 8B and 70B): on WildChat (GPT-4) 69K, padding / random packing / greedy packing averaged 49.58 / 49.46 / 50.6 at 8B and 61.50 / 65.97 / 65.92 at 70B; on a 1M open-source set, 54.3 / 54.95 / 55.05 at 8B and 66.12 / 67.26 / 67.54 at 70B; 70B training on WildChat took 9,533 s padded and 3,749 s with random packing (arXiv:2410.08081 Tables 3, 5).
- SmolLM3: packing raised throughput 3-5×; at the same effective batch of 128, IFEval dropped with packing, and above effective batch 32 the average fell for that model and dataset ([[smol-training-playbook]], "To Pack or Not to Pack?").

Two independent groups report worse results when a small SFT set is packed at the same epoch count: higher validation loss (Kundu et al., 20K examples) and lower IFEval (SmolLM3) (Replicated). Gains from packing at 70B and on the 1M-example set come from one study (Wang et al.; Result, single study).

**Position IDs.** Rotary position embeddings (RoPE) make the attention score depend on the offset between query and key positions, so a constant offset inside one correctly masked document leaves attention unchanged. Resetting `position_ids` still matters when the attention kernel computes `cu_seq_len` (the example boundaries) from them, as in Kundu et al. (§3.3), and for learned absolute positions (Krell et al., BERT, §4.2.1); it would also matter when un-reset positions exceed the trained range (Interpretation). In Kundu et al. Table 4 (Mistral-7B, FLAN 20K), packing without position IDs, which lets examples attend to each other, gave validation loss 1.294 against 1.221 with position IDs at the same 35 optimizer steps, and 1.252 against 1.170 at 281 steps. Wang et al. do not state that attention was reset between packed conversations.

**Implication.** When the SFT set is large (about 1M examples) or the model is large (70B), use packing, because in Wang et al. those runs trained 2.5 to 6.7 times faster (random or greedy packing) and changed the average score by −0.19 to +4.47 points (Tables 3, 5, all four datasets at 70B and the 1M set at 8B). When the model has 3B to 8B parameters and the set is small (20K examples in Kundu et al., 69K in Wang et al.; the SmolLM3 baseline subset sizes are not given in the text), compare packing against padding at matched optimizer steps, or use mini-batch packing, which kept the padded run's step count and validation loss in Kundu et al.

## §5 NEFTune

**Definition and formula** ([[neftune]] Algorithm 1). For embeddings X ∈ R^{B×L×d}, X′ = X + (α/√(L·d))·ε with ε ~ Uniform(−1, 1)^{B×L×d}. B is batch size, L sequence length, d embedding width, α the noise scale. Noise is added only during training.

**Evidence.** LLaMA-2 7B, GPT-4 judge against Text-Davinci-003: Alpaca 29.79 → 64.69, Evol-Instruct 70.34 → 79.60, ShareGPT 68.74 → 76.28, OpenPlatypus 62.00 → 70.61 (Table 1); ARC, HellaSwag, MMLU, and TruthfulQA stayed stable (Fig. 3). NEFT had higher training loss, slightly lower held-out loss, and lower ROUGE-L/BLEU against training responses (Figs. 4-5). It also lengthened outputs from 375 to 1,062 characters on Alpaca (Table 4). This matters because the headline metric is an LLM judge whose scores the authors report to be correlated with length: prompting LLaMA-1 7B (Alpaca) for longer answers raised its GPT-4 win rate from 32.36 to 48.01, below the 61.99 of the NEFT model (§5.2-5.3, Table 5).

**Relation to §2.** In Shi et al., NEFTune and IM each raised AlpacaEval 1.0 over IT on six of seven datasets, and adding NEFTune to IM raised AlpacaEval 1.0 over IM alone on six of seven datasets but lowered the NLP mean on LIMA, Less MMLU Chat, and Less BBH ICL (Tables 1, 4). Two independent groups report the same pattern (higher training loss, lower held-out loss, lower overlap with training outputs) for two different interventions; each method's pattern was measured by one group, so this is consistent evidence for the overfitting explanation rather than a replication of either method (Interpretation).

**Limits.** Tested on single-turn sets (Evol-Instruct 70k, ShareGPT 70K, Open-Platypus 25k; Alpaca size not stated), α chosen on AlpacaEval itself, safety not evaluated ([[neftune]] §3.2, App. A.1, §7). No source in the library tests NEFTune on a million-example multi-turn mix.

## §6 Template consistency between training and inference

**Definition.** A chat template is the rendering function from messages to token ids: role headers, turn terminators, and rules for thinking and tool content.

**Problem.** A template difference between SFT, later stages, and serving raises no error. Tülu 3 measured the effect of template choice on an intermediate SFT mix with Llama 3.0: Tülu with newline replaced by EOS 53.0, Zephyr 52.9, Tülu 3 without trailing newline 52.8, Tülu 2 52.6, Llama 3 51.6. The best option was not used "to avoid generation inconsistency with later steps in our post-training pipeline" ([[tulu-3]] §4.3.1, Table 13). Result (single study; the number of runs per template is not reported).

**Template design for mixed thinking modes.** Qwen3 fused thinking and non-thinking data in one SFT stage. Non-thinking samples keep an empty `<think></think>` block, which the authors describe as ensuring "internal format consistency", and multi-turn dialogs contain several `/think` and `/no_think` flags, with each response following the last flag ([[qwen-3]], arXiv:2505.09388v1 §4.3, Table 9; read at the paper because the card is not yet verified).

**Reported template and formatting-token problems in released pipelines.**
- OLMo 3 7B Think DPO and RL used a chat template slightly different from its SFT template because of "a minor miscommunication"; for OLMo 3.2+, Think SFT data is tokenized with the instruct template to work around a bug that masked the first `<think>` token as prompt ([[open-instruct-allenai-recipes]], `docs/olmo3.md` L22-28).
- SmolLM3's first hybrid baseline ignored personas in the system message; a bug in the processing code had set the template's `custom_instructions` argument to `None`, which removed the system message from every training sample. Fixing it did not change evaluation scores ([[smol-training-playbook]], "Vibe-Test Your Baselines").
- Llama 3 masks header and termination tokens from both chosen and rejected responses in DPO, because including them "may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens"; the authors hypothesize that shared tokens receive opposing gradients ([[llama-3]] §4.1.4).

**Checks.** Render one conversation with the training template and with the serving stack, compare token ids, and print the label mask per role. TRL obtains assistant masks from a `{% generation %}` block in the template with `return_assistant_tokens_mask=True` ([[smol-training-playbook]], "Masking User Turns"). **Implication.** Choice among five templates moved the Tülu 3 average by 1.4 points. No source in this chapter measures the score effect of a template mismatch between SFT and serving; the reported cases are pipeline bugs (OLMo 3, SmolLM3) and formatting-token behavior in DPO (Llama 3), so the rendered-token check above is the available control.

## §7 Agentic trajectories: reasoning history, erroneous steps, truncation, and token ids

**Reasoning history rules differ by model.**
- Qwen3: "the historical model output should only include the final output part and does not need to include the thinking content" (Qwen3-8B model card, Best Practices; [[chat-template-matrix]]).
- DeepSeek-V3.2: "Historical reasoning content is discarded only when a new user message is introduced"; with only tool messages appended, reasoning is retained, and tool calls and results are always kept (arXiv:2512.02556v1 §3.2.1; [[chat-template-matrix]]).
- MiniMax-M2 (official blog): retaining prior-round thinking against discarding it gave SWE-Bench Verified 69.4 vs 67.2, Tau² 87 vs 64, BrowseComp 44.0 vs 31.4, GAIA 75.7 vs 67.9, xBench 72.0 vs 66.0; evaluation settings are not given ([[minimax-m2-interleaved-thinking]]).
- GLM-5 trains "Preserved Thinking" for coding-agent scenarios, retaining all thinking blocks across turns ([[glm-5]] §3.1). SmolLM3 kept reasoning tokens in all turns during training because it judged this necessary "to condition the model appropriately" ([[smol-training-playbook]], "Picking a Good Chat Template").

The SFT consequence follows from §1.2. When the deployed client keeps or removes earlier thinking, render SFT histories with the same rule, because otherwise each trained turn is conditioned on a history the model does not receive at inference (Interpretation). A model trained with retained thinking and served through an API that drops it receives shorter histories than in training. MiniMax reports community deployments of M2 that did not preserve prior-round thinking and attributes this to APIs and clients that do not pass reasoning content back; its benchmark comparison above measures the cost of discarding that thinking (single official source, evaluation settings not reported).

**Erroneous steps.** GLM-5 keeps erroneous segments of agent trajectories in context "but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions" ([[glm-5]] §3.1). If the failed tool call were a training target, cross-entropy would raise its probability. GLM-5 reports no ablation of masking against discarding these trajectories.

**Truncation.** open-instruct documents that "right-sided truncation drops the trailing EOS, so a cut inside an assistant turn leaves trainable text with no terminator"; `keep` (default) trains it as is, `terminate` makes the last token a trainable EOS, and `drop` masks the row ([[loss-masking-regimes]], `dataset_transformation.py` L1427-1488). Truncation affects a larger share of rows when trajectories are long relative to `max_seq_length` (Interpretation; no source in this chapter reports the rate for agentic data).

**Token-in-token-out.** GLM-5 builds RL trajectories from the token ids the inference engine produced, because re-tokenizing text "can introduce subtle mismatches in token boundaries, whitespace/normalization handling, truncation, or special-token placement" ([[glm-5]] §4.1.2). When SFT data are taken from RL rollouts or rejection sampling, storing token ids avoids the same mismatch; open-instruct masks out and then drops conversations whose assistant label spans cannot be derived, about 0.005% of `tulu-3-sft-olmo-2-mixture` (`dataset_transformation.py` L1499-1504). The SFT use of token-in-token-out is this course's Interpretation; GLM-5 describes it for RL.

## §8 LoRA versus full fine-tuning: learning against forgetting

**Definition.** LoRA (low-rank adaptation) freezes a pretrained weight matrix W ∈ R^{d×k} and trains only a low-rank update, so the fine-tuned matrix is W + γ_r·AB with trainable A ∈ R^{d×r} and B ∈ R^{r×k} ([[lora-learns-less-forgets-less]] §2). Here r is the rank (r much smaller than d and k), γ_r = α/r is a scale factor, and α is a hyperparameter; Biderman et al. set α = 2r and apply LoRA to all attention and MLP matrices.

**Problem.** Full fine-tuning on a narrow domain can lower scores on tasks outside that domain (forgetting). LoRA restricts the update to rank r; Biderman et al. measure how much this reduces forgetting and how much target-domain learning it costs.

**Evidence.** Llama-2-7B on Magicoder-Evol-Instruct-110K (72.97M tokens), epoch 4: full fine-tuning HumanEval pass@1 0.470 with forgetting average 0.512; LoRA r = 256 0.498 with 0.631; LoRA r = 16 0.358 with 0.652. The forgetting average is the mean accuracy on HellaSwag, ARC-Challenge, and WinoGrande, so a higher value means less forgetting (Tables S5-S6, §3.3). Full fine-tuning also produced fewer unique HumanEval solutions out of 50 samples than LoRA, with the base model highest (§4.5, Fig. 5); the authors note that exact string matching is a coarse diversity measure. On the general Tülu-v2-mix, all LoRA ranks (16, 64, 256) were within one standard error of full fine-tuning on MT-Bench, and at 2 epochs full fine-tuning forgot less than LoRA; the main-paper ordering reappeared at epoch 6 (App. C.2). Thinking Machines (practitioner-evidence) found that LoRA matched full fine-tuning in log loss on Tulu3 and a subset of OpenThoughts3 until the adapter's capacity was exhausted, that the optimal LR was 10× higher for LoRA, and that attention-only LoRA was worse than MLP-only LoRA ([[lora-without-regret]]). Biderman et al. also report an optimal LoRA LR one order of magnitude above full fine-tuning (§4.7, App. B Fig. S1), so that ratio is Replicated.

**Conditions and limits.** Biderman et al. test one base model (Llama-2-7B), two target domains (code, math), and one general mix; Thinking Machines measures training log loss, not forgetting.

**Implication (Interpretation).** On code data, rank traded learning against forgetting; on the general Tülu-v2-mix at 2 epochs, the ordering of forgetting reversed. Forgetting therefore has to be measured for the data and epoch count in use, on suites outside the training mix; [[ch-30a]] covers the protocol and other controls.

## §9 Evaluation protocol for each design axis

Tülu 3 splits evaluation into a development suite used for decisions and an unseen suite not examined during development, for example IFEval (development) and IFEval-OOD (unseen) ([[tulu-3]] §2.2, Table 3). In its SFT ablations, removing Persona data lowered IFEval from 72.8 to 53.6 while IFEval-OOD moved from 17.6 to 18.0; removing math data lowered MATH from 31.5 to 23.5 and Deepmind Mathematics from 32.3 to 23.3 (Table 32). The authors conclude that their data choices "overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning" (§7.4.1); for math, Table 32 shows the development and unseen scores falling together when the math data are removed.

| Axis | Decide on | Confirm on before adopting | Noise to exceed |
|---|---|---|---|
| Loss over instructions / PLW | development tasks split into short-answer and long-generation | unseen suite; BLEU of outputs against training targets | seed spread (Tülu 3 Table 14) |
| Epochs, LR, aggregation | development average and per-skill scores | forgetting average against the base model; unseen suite | same |
| Packing | development scores at matched optimizer steps | unseen instruction-following task | same |
| Template and history rendering | rendered token ids equal in training and serving | agentic held-out tasks under the serving client | — |
| LoRA rank | target-domain score | forgetting average; unique generations per prompt | same |

## Negative samples and negative feedback

**Which meaning applies.** The course uses four meanings of "negative": (1) a sample with negative marginal value, removed before training; (2) a failure kept as content in the input or in a corrected target and trained with ordinary cross-entropy; (3) a failure trained under a control token such as "incorrect"; (4) an explicit decrease of a sample's likelihood (negative as gradient). Ordinary SFT uses (1) when it discards samples and (2) when it keeps failed steps in context with zero loss. This section also covers what SFT cross-entropy suppresses implicitly. Meaning (4) is covered in [[ch-31a]] and [[ch-39]].

**Where negatives come from in SFT.** Rejection sampling generates several outputs per prompt and keeps only the best or the correct ones; the rest are discarded. Llama 3 samples K outputs (typically 10 to 30) per prompt and keeps the one its reward model scores highest ([[llama-3-recipe]], v3 §4.2.2 row). Agentic trajectories contain failed tool calls, which GLM-5 keeps in context and masks from the loss ([[glm-5]] §3.1). Neither source reports a false-negative rate for these labels.

**Mechanism.** For logits z and target token y, softmax cross-entropy gives

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

where z_j is the logit of token j, p_j = softmax(z)_j is the model probability of token j, y is the target token, and 1[j = y] is 1 for the target and 0 otherwise. A gradient-ascent step on log p_y with learning rate η raises z_y by η(1 − p_y) and lowers every other logit by η·p_j, so the most likely wrong token at that position is pushed down the most. **Worked example:** p = (0.6, 0.3, 0.1), target is token 2, η = 1. Logits change by (−0.6, +0.7, −0.1); starting from z = log p, new probabilities are (0.322, 0.590, 0.088). The competitor at 0.6 lost 0.278, the one at 0.1 lost 0.012.

**What it cannot do.** The suppression acts only at positions of the dataset target under teacher forcing, where each position is conditioned on the dataset's previous tokens rather than on the model's own samples. It does not lower a specific wrong continuation the model would generate from its own earlier tokens, and it can raise the probability of bad responses that share tokens with good ones: in ORPO's OPT-350M experiment on HH-RLHF, SFT on chosen responses raised rejected-response log-probabilities along with chosen ones, and rejected ones were sometimes higher ([[orpo]] §3, Fig. 3). Adding λ·L_OR, an explicit decrease of the rejected response, reversed this at λ = 1.0 (Fig. 7).

**Controls and diagnostics.** Mask failed steps that stay in context; keep a set of known-bad responses and log their mean log-probability before and after SFT; report unique generations per prompt and pass@k (the probability that at least one of k samples is correct), since positive-only full fine-tuning reduced unique HumanEval solutions ([[lora-learns-less-forgets-less]] §4.5). **Effect on generality:** positive-only SFT on a narrow set concentrates probability on the demonstrated outputs; the measured cost in this chapter is lower diversity and higher forgetting, not a measured share of gain from negatives.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; schedule; warmup ratio | 5×10⁻⁶; linear; 0.03 | arXiv:2411.15124v5 Table 11, §4.3 [[tulu-3]] | verified 2026-09-15 | §4.3.2 Fig. 5: sum loss with 5e-6 best (Llama 3.0, Tülu 2 mix); "found after a hyperparameter search" |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | epochs; max length; effective batch (sequences) | 2; 4,096 tokens; 128 | Table 11 | verified 2026-09-15 | Fig. 6: 2 epochs best (Llama 3.0, Tülu 2 mix) |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | data; compute | 939,344 prompts; 32 GPUs, 6 hours | Table 7; §4.3 | verified 2026-09-15 | Fig. 4: full mix best among stratified subsamples |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss aggregation | paper: sum loss; open-instruct command `--reduce_loss sum` at 8781471, flag absent at 098424c (default mean) | §4.3.2; [[open-instruct-allenai-recipes]] | conflict | Figs. 5-6; the paper's value selected the LR |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | chat template; seed | Tülu 3 template without trailing newline (App. B.3); best single run of 5 seeds, seed 123 (60.1); the open-instruct command also passes seed 123 | Table 13; Table 14; [[open-instruct-allenai-recipes]] (docs/tulu3.md L53) | verified 2026-09-15 | Table 13 (runs per template not reported); Table 14 seeds 59.8-60.1 |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | peak LR; epochs; batch; length; warmup; seed | 2×10⁻⁶; 2; 128; 4,096; 0.03; best of 3 seeds, seed 456 (72.6) | Table 11; §4.3; Table 14 | verified 2026-09-15 | hyperparameter search; Table 14 seeds 70.0-72.6 |
| Llama-3.1-Tulu-3-8B/70B-SFT | 8B, 70B | SFT | packing; NEFTune; optimizer betas | not reported | checked §4.3, Table 11, App. B | not reported | — |
| open-instruct SFT default (not a released model) | any | SFT | loss mask | all assistant turns (`last_turn_only=False`) | open-instruct@098424c `finetune.py` L132-133, `dataset_transformation.py` L1214-1218 | verified 2026-09-15 (code only) | no ablation reported |
| zephyr-7b-sft-full (alignment-handbook recipe file) | 7B | SFT | LR; schedule; warmup ratio; epochs; max length | 2.0e-05; cosine; 0.1; 1; 2048 | alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml` [[hf-alignment-handbook]] | verified 2026-09-15 | no ablation reported; the file at this commit may differ from the 2023 run |
| zephyr-7b-sft-full | 7B | SFT | per-device batch; accumulation; data; packing/mask keys | 16; 1; ultrachat_200k `train_sft` and `test_sft`, weight 1.0 each; no packing, assistant-only, or NEFTune key | same file | verified 2026-09-15 | no ablation reported |
| zephyr-7b-sft-full (Hub model card, generated by the Trainer; Transformers 4.36.2) | 7B | SFT | LR; devices; total batch (sequences); optimizer; schedule; epochs; steps | 2e-05; 8; 128 (16 × 8); Adam β (0.9, 0.999), ε 1e-08; cosine, warmup ratio 0.1; 1; 1,090 | huggingface.co/alignment-handbook/zephyr-7b-sft-full README, "Training hyperparameters" and "Training results" | verified 2026-09-15 | no ablation reported; agrees with the config file on LR, schedule, warmup, epochs, per-device batch |
| Llama 3 "largest models" (size not printed at this locus) | not printed | SFT | LR; steps; loss | 10⁻⁵; 8.5K-9K steps; cross-entropy on target tokens, prompt tokens masked | arXiv:2407.21783v3 §4.1.3 [[llama-3-recipe]] | verified 2026-09-14 | "work well across different rounds and data mixes"; no numbers |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | SFT | epochs; length; LR; weight decay; clipping | 2; 32,768; 7×10⁻⁶ → 7×10⁻⁷; 0.1; 1.0 | arXiv:2412.15115v2 §4.1 [[qwen-2.5-recipe]] | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev1 and Dev3 SFT) | 671B MoE | SFT | base; epochs; LR; context; batch | DeepSeek-V3-Base; 2-3; cosine 5×10⁻⁵ → 5×10⁻⁶; 32,768; 128 | arXiv:2501.12948 v2 B.4.2 [[deepseek-r1-recipe]] | verified 2026-09-14 | no ablation reported |
| SmolLM3-3B SFT ablation baseline | 3B | SFT | LR; effective batch; epochs; packing; max length | 1×10⁻⁵; 128; 1; off; 8,192 (Instruct) or 32,768 | Smol Training Playbook, "Baby Baselines" [[smol-training-playbook]] | verified 2026-09-15 | LR scan: 3e-6 or 1e-5 above larger LRs; values only in figures |
| LLaMA-2-7B IT/IM runs (Shi et al.) | 7B | SFT | LR; batch; epochs; length; AdamW β, ε; schedule; weight decay | 2×10⁻⁵; 128; typically 2 (2, 3, or 10); 2,048; (0.9, 0.98), 1e-6; linear, warmup 0.03; 0 | arXiv:2405.14394v2 App. C, Table 6 [[loss-masking-prompt]] | verified 2026-09-15 | no ablation reported |
| LLaMA-2 7B NEFTune runs | 7B | SFT | LR; epochs; batch; length; α | 5e-5 (Adam); 3; 128; 512; α 5, 5, 10, 15 (Alpaca, Evol, ShareGPT, OpenPlatypus) | arXiv:2310.05914v2 App. A.1, Table 7 [[neftune]] | verified 2026-09-14 | LR: initial sweep on LLaMA-1 7B with Alpaca (A.1); α: best of {5, 10, 15} on AlpacaEval with a ChatGPT judge (A.1); epochs: no selection procedure reported; the App. A.2 epoch ablation (Table 14) peaks at 3 with NEFT |
| Llama-2-7B on Tülu-v2-mix (Biderman et al.) | 7B | SFT (full, LoRA) | LR; optimizer; batch; length | full 5e-6, LoRA 1e-4 (r = 16, 64, 256, α = 2r); decoupled LionW (0.9, 0.95); 192; 4,096 | arXiv:2405.09673v2 App. C.1 [[lora-learns-less-forgets-less]] | verified 2026-09-15 | "After an initial learning rate sweep"; App. B Fig. S1 |

Starting point for a small general-purpose run: for a base model of about 8B parameters and a general mix near one million examples, the verified Tülu 3 8B rows give LR 5×10⁻⁶ with linear schedule and warmup ratio 0.03, 2 epochs, effective batch 128 sequences, maximum length 4,096 tokens, and a sum loss; Tülu 3 used these on Llama 3.1 8B with 939,344 prompts on 32 H100 GPUs for 6 hours. With a token-mean loss the LR is not transferable and needs a sweep. For a 3B model and smaller ablation mixes, the SmolLM3 baseline used LR 1×10⁻⁵, effective batch 128, 1 epoch, and no packing, on one node of eight H100 GPUs for 30 to 90 minutes per subset. Repeat any close comparison across seeds, since Tülu 3 SFT seeds spread 0.3 average points at 8B and 2.6 points at 70B.

## Generalization lens

**(a) What increases breadth.** Loss over instructions raised the 18-task NLP mean on all seven small datasets in Table 1 and AlpacaEval 1.0 on six of them ([[loss-masking-prompt]] Table 1). NEFTune raised AlpacaEval while ARC, HellaSwag, MMLU, and TruthfulQA stayed stable ([[neftune]] Fig. 3). The Tülu 3 math data raised both the development and the unseen math score ([[tulu-3]] Table 32). Packing raised the average score at 70B on WildChat (GPT-4) 69K from 61.50 to 65.97 (random) or 65.92 (greedy), and on the 1M-example set from 54.3 to 54.95 or 55.05 at 8B and from 66.12 to 67.26 or 67.54 at 70B ([[sequence-packing-contract]], Wang et al. Table 3; single study, no seeds reported).

**(b) What causes narrowing or forgetting.** Instruction tuning on 9,229 Dolly examples lowered the 18-task mean from the base model's 49.32 to 45.54 ([[loss-masking-prompt]] Table 3), and IT "generally" had a higher instruction tuning tax than IM across epochs 2-10 (Fig. 4). Full fine-tuning on code forgot more than LoRA and produced fewer distinct solutions ([[lora-learns-less-forgets-less]] Tables S5-S6, Fig. 5). Tülu 3 Persona data raised IFEval by 19.2 points while IFEval-OOD was 0.4 points lower with it ([[tulu-3]] Table 32). Packing the SmolLM3 baseline set at the same effective batch size of 128 lowered IFEval ([[smol-training-playbook]], "To Pack or Not to Pack?"; no numbers in the text). A KL penalty to the base model protected NLP tasks but reduced AlpacaEval 2.0 to near the base model's 0.01 ([[loss-masking-prompt]] Table 3).

**(c) How to measure it at this stage.** Use a development suite for decisions and an unseen suite for reporting ([[tulu-3]] §2.2). Report a forgetting average against the base model on tasks absent from the mix ([[lora-learns-less-forgets-less]] §3.3). Report training and held-out loss on output tokens and BLEU between outputs and training targets as overfitting probes ([[loss-masking-prompt]] Fig. 3, Table 2). Report unique generations per prompt. Compare configurations only when the difference exceeds the seed spread ([[tulu-3]] Table 14). Known measurement errors: LLM-judge win rates rise with output length, and prompting for longer outputs alone raised one AlpacaEval win rate from 32.36 to 48.01 ([[neftune]] §5.2-5.3, Table 5); a single development benchmark can be overfit (IFEval above); exact-string uniqueness is a coarse diversity measure ([[lora-learns-less-forgets-less]] §4.5).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Assuming a conversational dataset is assistant-only by default in TRL | labels ≠ −100 on user tokens; the model is trained to continue user queries (SmolLM3 "Masking User Turns") | print labels per role; set `assistant_only_loss=True` and confirm masks |
| Training last turn only by accident | fewer trained tokens than assistant tokens | count labels ≠ −100 against assistant token count |
| Copying a sum-loss LR into a mean-loss trainer | results change when GPU count or accumulation changes | run two accumulation settings at fixed total batch and compare |
| Training thinking history that the serving client drops | agentic held-out scores lower under the real client (MiniMax-M2 comparison in §7; training rendering not reported there) | diff rendered training prefix at step k with the client request |
| Packing a small set with unpacked epochs and LR | steps per epoch fall by the pack factor; IFEval drops | log steps and tokens per step; compare at matched steps |
| Padding-free collation without `position_ids` resets, in a kernel that derives boundaries from them | examples attend across boundaries; Mistral-7B on FLAN 20K: validation loss 1.294 against 1.221 with position IDs at the same 35 optimizer steps (Kundu Table 4); across the 10 models of Kundu Table 2, where the two rows also differ in row count, loss without position IDs minus loss with them ranged from −0.089 (Phi-2) to +0.170 (Falcon) | assert boundaries derived from `position_ids` equal example lengths |
| Leaving failed tool calls as trainable targets | label audit shows trained tokens inside error spans; no source in this chapter measures the downstream effect | label audit on error turns; mask erroneous segments (GLM-5 §3.1) |
| Truncating with `keep` on long trajectories | unterminated responses trained | count truncated rows; choose `terminate` or `drop` |
| Selecting epochs or data on the reporting benchmark | development gain without held-out gain | keep an unseen suite unexamined until the end |
| Using full fine-tuning LR for LoRA | LoRA underperforms full fine-tuning | sweep; the reported optimum is about 10× higher for LoRA |
| Deciding from one seed | ranking flips on rerun | train several seeds; compare with the Table 14 spread |

## Check your understanding

1. Explain why adding loss on instruction tokens can lower held-out loss on completions even though the added targets are not completions. Which measurements in Shi et al. support your explanation, and which of their datasets contradict a uniform benefit?
2. Using the prompt-loss-weight worked example, explain why the best PLW depends on the completion/prompt length ratio and on whether the benchmark requires long or short generations.
3. Two runs use the same LR and data but different gradient accumulation. Explain, with the Tülu 3 equations, which samples each run weights more and why a sum loss removes the difference.
4. A team packs a 20K-example SFT set and sees higher validation loss than the padded run. Give two causes that predict different fixes, and the measurement that separates them.
5. Explain when single-pass loss on all assistant turns and per-turn expansion give different gradients, using the Qwen3 and DeepSeek-V3.2 history rules.
6. Why does ordinary SFT fail to lower the probability of a wrong tool call that the model itself generates, even though cross-entropy lowers all non-target logits?
7. Tülu 3 Persona data raised IFEval by 19.2 points and did not raise IFEval-OOD. What does this imply for how SFT settings should be selected and reported?
8. LoRA forgot less than full fine-tuning on code instruction data, but on Tülu-v2-mix at 2 epochs full fine-tuning forgot less. What properties of the data and training length could explain the difference, and how would you test them?

## Connections

- Previous: **ch-29e — Instruction Tuning and Generalization to Unseen Tasks** ([[ch-29e]]): data properties that give unseen-task generalization; this chapter varies training settings on such data.
- Dependency: **ch-04 — Sequence Packing, Loss Masking, and Chat Templates** ([[ch-04]]): mechanics of `cu_seqlens`, masks, and templates.
- Next: **ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control** ([[ch-30a]]): the forgetting report and controls referenced in §8.
- **ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares** ([[ch-30b]]); **ch-30c — Weight Averaging and Model Merging for Generalist Models** ([[ch-30c]]).
- **ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood** ([[ch-31a]]); **ch-39 — Offline Preference Optimization: DPO and Its Variants** ([[ch-39]]).
- **ch-26 — Tool and Function-Calling Data** ([[ch-26]]); **ch-27 — Agentic Trajectory Data** ([[ch-27]]).
- **ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split** ([[ch-36]]): runs the axes of this chapter.

## Sources

- [[loss-masking-prompt]] — Shi et al. 2024 (arXiv:2405.14394v2): IM definition, Tables 1-5, Figs. 2-4, App. C settings. The library card still states the opposite result and needs correction.
- [[loss-masking-regimes]] — excerpt: Huerta-Enochian and Ko 2024 PLW results; open-instruct@098424c masking and truncation code; TRL v0.23.0 loss options.
- [[neftune]] — Jain et al. 2023: noise formula, Table 1, overfitting and length analyses, settings.
- [[sequence-packing]] — Krell et al. 2021: padding fractions and BERT position-ID ablation.
- [[sequence-packing-contract]] — excerpt: Krell, Kundu et al. 2024, and Wang et al. 2024 packing results.
- [[hf-alignment-handbook]] — Zephyr-7B-β SFT config at alignment-handbook@1de1fc9 and the Trainer-generated card of `alignment-handbook/zephyr-7b-sft-full` (card itself unverified).
- [[tulu-3]] — Tülu 3 report v5: Tables 3, 7, 11, 13, 14, 32; §4.3.2 aggregation (card itself unverified; values read from the paper).
- [[tulu-3-sft-recipe]] — excerpt of the Tülu 3 SFT settings and ablations as printed in the paper.
- [[open-instruct-allenai-recipes]] — reproduction commands, loss-reduction drift, OLMo 3 template notes.
- [[chat-template-matrix]] — excerpt: Tülu 3, Zephyr, Llama 3 template facts; Qwen3 model card and DeepSeek-V3.2 history rules.
- [[smol-training-playbook]] — excerpt: SmolLM3 SFT baselines, masking, packing, LR, epochs, template design.
- [[lora-learns-less-forgets-less]] — excerpt: Biderman et al. 2024 learning and forgetting tables.
- [[lora-without-regret]] — excerpt: Thinking Machines 2025 LoRA capacity and LR results.
- [[minimax-m2-interleaved-thinking]] — excerpt: retained against discarded thinking results.
- [[glm-5]] — excerpt: preserved thinking, masked erroneous segments, token-in-token-out.
- [[orpo]] — SFT raises rejected log-probabilities (Fig. 3); odds-ratio term (Fig. 7).
- [[qwen-3]] — Qwen3 report arXiv:2505.09388v1 §4.3, Table 9: empty think block and mode flags in the thinking-mode-fusion SFT template (card itself unverified; read at the paper).
- [[llama-3]], [[llama-3-recipe]] — chat protocol, SFT loss and LR, DPO formatting-token masking.
- [[qwen-2.5-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]] — SFT recipe rows; R1-Zero RL without SFT.
- [[in2-film]] — Mistral `[INST]` template in a Mistral-based SFT run.
- [[allenai-tulu-sft-recipe]] — cited only in Corrections; its NEFTune, packing, and optimizer rows are not in the Tülu 3 report.
