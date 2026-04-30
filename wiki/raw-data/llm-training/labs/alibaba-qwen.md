<!-- scope: Alibaba Qwen team contributions to LLM post-training -->

# Alibaba Qwen
- **Core Contribution:** Broad open-weight release cadence (0.5B -> 72B, dense + MoE, vision + coder + math) with SFT + DPO + GRPO post-training and rich long-context training.
- **URL:** https://qwen.ai/ | https://github.com/QwenLM
- **Key people:** Jinze Bai, Junyang Lin, Xuancheng Ren, and the Qwen team (Alibaba Cloud / Tongyi)

## Signature artifacts
- [[qwen-2.5]] — Qwen 2.5 technical report (Dec 2024) — SFT + DPO with Online Merging Optimizer + GRPO with variance-prioritized prompts.
- [[qwen-3]] — 2025 follow-up with unified reasoning / non-reasoning mode.
- Qwen2.5-Math (arXiv 2409.12122) — self-improvement math post-training.
- Qwen2-VL / Qwen2-Audio — multimodal extensions.
- Qwen-Coder family.
- Qwen2.5-MoE — MoE variant joining DeepSeek-V3/Mixtral in the open MoE space.

## Position in the field
Qwen is the widest-scope open-weight LLM family of 2024–2025 — more size points, more modality coverage, and more frequent releases than any other lab. On post-training they adopt the standard SFT -> DPO -> RL stack but differentiate with (a) a two-stage SFT context curriculum (short prompts first, then a mixed short+long phase up to 262K tokens), (b) the Online Merging Optimizer as a DPO stabilizer, and (c) variance-ordered GRPO — prioritizing prompts whose responses the reward model scores with high variance.

Qwen is particularly strong on Chinese + English dual capability and on math (Qwen2.5-Math-72B matches much larger models via self-improvement). Qwen is also frequently used as the base for distilled R1 students, making Qwen a de-facto foundation for the Chinese open-RL ecosystem.

## Anticipated future work
- Qwen3 continues scaling with reasoning-mode toggle (explicit <think> mode).
- Continued integration of agentic tool use following Kimi K2's lead.
- MoE lineage expected to scale further toward DeepSeek-V3-class total parameter counts.
- Possible Qwen-reasoning dedicated model competing with R1.

## Related pages
- [[qwen-2.5]], [[qwen-3]], [[deepseekmath]] (algorithmic cousin), [[grpo]].
- [[deepseek]] — main domestic competitor; contrast in MoE routing and RL scope.
- [[tulu-3]] — spiritual open-recipe cousin using similar three-stage pipeline.
