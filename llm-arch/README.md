# LLM Architecture — Reference Library

Source materials for the **LLM Architecture: Foundations to Frontier** course.
Materials are organized by type and referenced during each chapter's read phase.

## Structure

```
llm-arch/
├── papers/          # Academic papers — abstracts, key contributions, architecture details
├── model-reports/   # Technical reports for specific models (LLaMA, DeepSeek, etc.)
├── blogs/           # Blog posts, explainers, course syllabi
├── urls.txt         # User-curated source URLs
└── *.pdf            # Original PDFs for offline reading
```

## User-Provided Sources

| Source | Type | Relevant Chapters |
|--------|------|-------------------|
| [Sebastian Raschka's Magazine](https://magazine.sebastianraschka.com/) | Blog | All |
| [HuggingFace Daily Papers](https://huggingface.co/papers/date/2026-04-20) | Paper feed | All |
| [Berkeley RDI Adv. LLM Agents (SP25)](https://rdi.berkeley.edu/adv-llm-agents/sp25) | Course | Ch 28-29 |
| [Berkeley RDI LLM Agents (F24)](https://rdi.berkeley.edu/llm-agents/f24) | Course | Ch 28-29 |
| [HuggingFace nanotron](https://huggingface.co/nanotron) | Framework | Ch 13 |
| [OpenMythos](https://github.com/kyegomez/OpenMythos) | Repo | Advanced |
| Ultra-Scale Playbook (PDF) | Book | Ch 11, 13, 25 |

## Chapter → Reference Mapping

| Chapter | Key References |
|---------|---------------|
| Ch 1: Language Modeling | pitfalls-next-token, raschka-next-token-prediction, hf-perplexity |
| Ch 2: Attention | attention-is-all-you-need, alammar-illustrated-transformer |
| Ch 3: The Transformer | attention-is-all-you-need, alammar-illustrated-transformer |
| Ch 4: Decoder-Only | gpt-1, gpt-2, gpt-3, bert, alammar-illustrated-gpt2 |
| Ch 5: Tokenization | (to be added) |
| Ch 6: Positional Encoding | rope, alibi, (iRoPE via llama-4 report) |
| Ch 7: Attention Variants | mqa, gqa, flash-attention, flash-attention-2 |
| Ch 8: FFN & Activations | glu-variants |
| Ch 9: Normalization | rmsnorm |
| Ch 10: Scaling Laws | scaling-laws-kaplan, chinchilla |
| Ch 11: Pre-training | ultra-scale-playbook |
| Ch 12: Post-training | deepseek-r1, (DPO paper to be added) |
| Ch 13: Distributed Training | ultra-scale-playbook, hf-nanotron |
| Ch 14: MoE | switch-transformer |
| Ch 15: SSMs | mamba, mamba-2 |
| Ch 16: Long Context | yarn, rope, alibi |
| Ch 17: Multimodal | (to be added) |
| Ch 18: LLaMA | llama-1, llama-2, llama-3, llama-4 |
| Ch 19: DeepSeek | deepseek-v2, deepseek-v3 |
| Ch 20: Gemma | gemma-3 |
| Ch 21: Jamba | jamba |
| Ch 22: Mamba-2 | mamba, mamba-2 |
| Ch 23: Qwen | qwen-3 |
| Ch 24: OLMo | olmo-2 |
| Ch 25-27: Inference | paged-attention, ultra-scale-playbook |
| Ch 28-29: Research | berkeley-adv-llm-agents-sp25, berkeley-llm-agents-f24 |
