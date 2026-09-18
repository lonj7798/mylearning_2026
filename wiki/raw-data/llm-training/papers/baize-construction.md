<!-- scope: Baize (Xu et al., EMNLP 2023): ChatGPT self-chat corpus construction, LoRA SFT of LLaMA, and Self-Distillation with Feedback (SDF)
     deps: [[self-instruct]]
     see-also: [[ultrachat-pipeline]], [[camel]], [[soda]], [[openassistant]], [[alpaca]], [[baize-construction-recipe]]
-->

# Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data
- **Core Insight:** One gpt-3.5-turbo call per dialogue that writes both the human and the AI turns around a seed question produced 111.5k dialogues for about $100 (§3), and LoRA-tuned LLaMA models trained on self-chat data and refined with SDF scored 90% (7B) and 92% (13B) of ChatGPT's GPT-4-judged score on the 80 Vicuna prompts (Fig. 3).
- **Guideline:** When multi-turn data is synthesized from an API model and budget allows, generate the AI turns one call at a time, because the paper reports that single-call self-chat yields shorter AI turns (average length 35.9 for Quora vs 149.6 for Quora v2, Table 2) and used per-turn responses for its later v1.5 data; otherwise the single-call template is the lower-cost option (§3).
- **Authors:** Canwen Xu, Daya Guo, Nan Duan, Julian McAuley (UC San Diego; Sun Yat-sen University; Microsoft Research Asia)
- **Year:** 2023 (arXiv v1 2023-04; EMNLP 2023)
- **URL:** https://arxiv.org/abs/2304.01196
- **Source type:** paper
- **Relevant topics:** self-chat, multi-turn dialogue synthesis, distillation from ChatGPT, LoRA, Self-Distillation with Feedback, loss masking, LLM-judge bias

## Abstract
Chat models such as ChatGPT are available only through restricted APIs. The paper proposes a pipeline that generates a multi-turn chat corpus automatically by having ChatGPT converse with itself. It then applies parameter-efficient tuning to the open LLaMA model. The resulting model, Baize, performs well in multi-turn dialogue and has guardrails that reduce potential risks. The paper also proposes Self-Distillation with Feedback (SDF), which further improves Baize with feedback from ChatGPT. The models and data are released for research use only.

## Key Contributions
- A self-chat pipeline: one template makes ChatGPT write both sides of a dialogue around a seed (a question or key phrase) until a natural stopping point (§3; App. A).
- Domain control through the seed set: Quora and Stack Overflow questions for general chat, MedQuAD for a healthcare model (§3).
- LoRA applied to all linear layers of LLaMA-7B/13B/30B to increase trainable parameters (§4; Table 3).
- SDF: Baize v1.5 samples four responses per Quora question, ChatGPT scores them, and the best one trains new LoRA modules, giving Baize v2 (§4; Fig. 2; App. C).
- Release of weights (research use only) and the corpus under CC-BY-NC 4.0 (Limitations).

## Key Figures/Tables to Study
- **Fig. 1** — pipeline for Baize and Baize v2. **Fig. 2** — SDF ranking example.
- **Table 1** — a not cherry-picked self-chat transcript from a Quora seed.
- **Table 2** — dialogues, average turns, and average response length per corpus.
- **Table 3** — base model, trainable parameters, GPU hours, and data for each Baize model.
- **Fig. 3** — GPT-4 score relative to ChatGPT. **Table 7** — Open LLM Leaderboard results at 13B.
- **App. A / App. C** — self-chat template and SDF feedback prompt.

## Technical Details
### Self-chat corpus
- Teacher model: ChatGPT (`gpt-3.5-turbo`) (§3). The API generates the transcript for both sides until a natural stopping point (§3).
- Self-chat template (App. A, verbatim):
  ```
  Forget the instruction you have previously received. The following is a conversation between a human and an AI assistant. The human and the AI assistant take turns chatting about the topic: '${SEED}'. Human statements start with [Human] and AI assistant statements start with [AI]. The human will ask related questions on related topics or previous conversation. The human will stop the conversation when they have no more question. The AI assistant tries not to ask questions. Complete the transcript in exactly that format.
  [Human] Hello!
  [AI] Hi! How can I help you?
  ```
- Baize v1 corpus: 111.5k dialogues, using about 55k questions from each of Quora and Stack Overflow; API cost approximately $100 (§3). MedQuAD seeds add 47k medical dialogues for Baize-Healthcare (§3).
- Table 2 (dialogues / average turns / average response length per turn; the length unit is not stated): Alpaca 51,942 / 1.0 / 44.2; Quora 54,456 / 3.9 / 35.9; StackOverflow 57,046 / 3.6 / 36.0; MedQuAD 46,867 / 3.8 / 35.8; Quora v2 55,770 / 3.0 / 149.6; StackOverflow v2 112,343 / 3.9 / 78.2 (Table 2).
- Single-call generation produces shorter AI turns than asking ChatGPT one turn at a time, but per-turn calls cost more because the context is attached multiple times. For the v1.5 data, a second ChatGPT generated each AI response one at a time and replaced the AI turns in the self-chat transcript (§3).
- Alpaca's single-turn data is added to the v1 training mix "to further enhance the ability of Baize to follow instructions" (§3; Table 3).
- The authors contrast self-chat with ShareGPT dialogues (used by Vicuna), which may contain sensitive personal information and complex copyright; self-chat avoids third-party copyright concerns if the seed dataset has a proper license (§3).

### Training and SDF
- LoRA (§4, Eq. 1): `h = W0 x + B_sft A_sft x`. Here `x` is the layer input, `h` the output, `W0 ∈ R^{d×k}` the frozen pretrained weight, `B_sft ∈ R^{d×r}` and `A_sft ∈ R^{r×k}` the trainable matrices, and `r ≪ min(d, k)` the rank. Only `A_sft` and `B_sft` are updated, and LoRA is applied to all linear layers (§4).
- For Baize v1.5, loss is computed only on the AI responses, following Vicuna (§4).
- SDF (§4, Eq. 2): `h = W0 x + B_sft A_sft x + B_sdf A_sdf x`, where `B_sdf ∈ R^{d×r}` and `A_sdf ∈ R^{r×k}` are new LoRA matrices; only these are updated during SDF.
- SDF steps: (1) Baize v1.5 generates four responses for each instruction from the Quora dataset; (2) ChatGPT gives each response an overall score from 1 to 100 for helpfulness, relevance, accuracy, and level of detail; (3) the best-ranked response is used to fine-tune the new LoRA modules (§4; App. C).
- The authors state that SDF needs no reward model, is 3× faster than RLHF with PPO, and, because it distills the model's own generations, has lower loss and avoids "possible catastrophic forgetting" (§4). No measurement supporting these statements is reported.
- Inference uses a prompt with `[|Human|]` / `[|AI|]` tags and the rule that the assistant declines unethical, controversial, or sensitive topics (§5; App. B). Decoding default: nucleus sampling, temperature 1, top-p 0.95 (§5).

### Evaluation
- GPT-4 score: Vicuna's pipeline with 80 hand-crafted prompts in 9 categories, scored relative to ChatGPT; ChatGPT's answer is placed first because GPT-4 prefers the first answer (§6). Scores: LLaMA-13B 68%, Alpaca-13B 76%, Baize-v1.5-7B 87%, Vicuna-7B 88%, Baize-v1.5-13B 89%, Baize-v2-7B 90%, Vicuna-13B 92%, Baize-v2-13B 92%, ChatGPT 100% (Fig. 3).
- Open LLM Leaderboard, Baize-v2-13B vs LLaMA-13B: ARC (25-shot) 50.3 vs 50.8; HellaSwag (10-shot) 77.1 vs 78.9; MMLU (5-shot) 39.4 vs 37.7; TruthfulQA (0-shot) 48.3 vs 39.9; average 53.8 vs 51.8. Vicuna-13B averages 53.7 and Alpaca-13B 51.7. Only the 13B result was available (Table 7).
- Falcon-40B-instruct, ranked #1 on that leaderboard as of June 23, 2023, was also fine-tuned with Baize data (§6).
- Qualitative examples are labeled cherry-picked or not; one healthcare practitioner confirmed the Baize-Healthcare responses in Table 9 (§6).
- The authors found that GPT-4 as judge prefers longer responses and has a positional bias (Limitations).

## Recipe ledger
The paper discloses LoRA SFT and SDF settings per model (§4-§5; Table 3). The full ledger is in [[baize-construction-recipe]].

## Findings relevant to generality, negative feedback, and distillation
- **Distillation.** All dialogue data is ChatGPT output, and Baize learns its guardrails by imitating ChatGPT plus the inference prompt; changing the prompt can remove the guardrails (Limitations).
- **Negative samples (discard).** SDF scores four sampled responses and trains only on the best-ranked one; the paper describes no use of the lower-scored responses, so they are discarded rather than used as content or as gradient (§4; App. C). Moving from v1.5 to v2 (SDF) changed the GPT-4 score from 87% to 90% at 7B and from 89% to 92% at 13B (Fig. 3). Result (single study; one judge; 80 prompts).
- **Breadth after SFT.** Relative to LLaMA-13B, Baize-v2-13B changes MMLU by +1.7, TruthfulQA by +8.4, HellaSwag by -1.8, and ARC by -0.5 points (derived from Table 7). The SDF "no forgetting" statement is not tested (§4).
- **Dialogue length.** Self-chat dialogues average 3.0-3.9 turns (Table 2), and training uses a maximum input length of 512 tokens for v1 and 1024 for v2 (§5). The paper reports no analysis of dependencies across turns.
- **Stated future work.** Diversify simulated user queries and improve self-chat quality (§7).

## Connections
- [[self-instruct]] / [[alpaca]] — Alpaca's Self-Instruct data is single-turn (§3); its 51,942 examples are part of the v1 mix, and Alpaca-13B is a baseline (Table 2; Fig. 3; Table 7).
- [[ultrachat-pipeline]] — a later ChatGPT-based multi-turn dialogue synthesis pipeline.
- [[camel]] — role-playing agents as another synthetic-dialogue method.
- [[soda]] — dialogue distillation grounded in social commonsense.
- [[openassistant]] — human-written conversation data.
- [[baize]] — duplicate card that redirects here.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2304.01196 (arXiv v4, 2023-12-02; "Baize v2; EMNLP 2023").
- Corrections to the previous card version:
  - Title "Baize (Construction): Self-Chat Protocol ..." → exact paper title.
  - Template shown with `[|Human|]` / `[|AI|]` and "turn cap (~8 turns)" → self-chat template uses `[Human]` / `[AI]` and runs until a natural stopping point (App. A; §3); `[|Human|]` belongs to the inference prompt (App. B).
  - "111.5K seeds from Quora, StackOverflow-Medical, Alpaca, and Medical-QA ... 111.5K dialogs (each 4+ turns)" → 111.5k dialogues from about 55k Quora and 55k Stack Overflow questions; Alpaca is single-turn data added to training; MedQuAD is a separate 47k set; average turns 3.6-3.9 (§3; Table 2).
  - "avg 4.5 turns, ~100 tokens per turn" → 3.9 / 3.6 turns and average response length 35.9 / 36.0, unit not stated (Table 2).
  - "Cost ~$1,000" → approximately $100 (§3).
- Removed as unsupported by the source: filtering steps (length check, n-gram repetition check, API-error markers); "median 4, tail to 10" turn distribution and "OASST, WildChat 15+"; "Baize-7B matches Alpaca-7B on MMLU; outperforms on MT-Bench"; "Baize-13B preferred over Alpaca-13B 58%"; "ROUGE-L between seed questions ≤ 0.15"; "orders-of-magnitude lower cost than crowdsourcing"; "first widely-adopted method"; "ancestor of UltraChat, OpenHermes"; self-chat failure modes (leading user questions, verbosity tics); "superseded in quality by UltraChat and OpenAssistant".
- Not reported by the source: any filtering or deduplication of self-chat output; epochs; learning-rate schedule; decoding settings used for SDF candidates; quantitative healthcare evaluation.
