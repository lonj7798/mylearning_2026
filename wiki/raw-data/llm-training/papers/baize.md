<!-- scope: redirect — duplicate card for Baize (arXiv:2304.01196); the full card is [[baize-construction]]
     see-also: [[baize-construction]], [[baize-construction-recipe]]
-->

# Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data
- **Authors:** Canwen Xu, Daya Guo, Nan Duan, Julian McAuley
- **Year:** 2023 (arXiv v1 2023-04; EMNLP 2023)
- **URL:** https://arxiv.org/abs/2304.01196
- **Source type:** paper

> Duplicate of [[baize-construction]]. This file is kept so existing links resolve.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2304.01196 (arXiv v4, 2023-12-02).
- Corrections to the previous card version (all now stated correctly in [[baize-construction]]):
  - "100K self-chat dialogues across Quora / StackOverflow / Alpaca / medical" → 111.5k self-chat dialogues from Quora and Stack Overflow seeds; Alpaca (51,942) is single-turn data added to training; MedQuAD adds a separate 47k set (§3; Table 2).
  - Prompt "Please simulate a conversation between a user and an AI assistant ..." with `[User]` / `[AI]` tags → the App. A self-chat template with `[Human]` / `[AI]` tags.
  - SDF described as "ChatGPT rate / critique / rewrite ... use as SFT or DPO pairs ... iterate" → Baize v1.5 samples 4 responses per Quora question, ChatGPT scores each from 1 to 100, and only the best-ranked response trains new LoRA modules; one round is described; no DPO (§4; App. C).
  - Authors' affiliations "UCSD + MSRA" → UC San Diego, Sun Yat-sen University, Microsoft Research Asia (title page).
- Removed as unsupported by the source: filtering rule "drop dialogues <3 turns, malformed, looping"; "among the first viable open multi-turn chatbots"; "most open models were single-turn only"; "research-use-only because of OpenAI TOS" (the paper states research-use weights following Alpaca and a CC-BY-NC 4.0 corpus); "ultrachat-pipeline (200K+ dialogues)"; "integrated into openhermes catalogues"; "first widely-reproduced open multi-turn SFT corpus"; the rlcd connection.
