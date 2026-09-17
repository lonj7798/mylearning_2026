<!-- excerpt for ch-20 §7: verbatim license and output-use statements from official model pages
     created: 2026-09-15 (generality revision)
     used-by: [[read]] §7
-->

# Excerpt: teacher-model license statements quoted from official pages

This file quotes the exact statements that ch-20 §7 relies on. It is not legal advice. A license file
can change after the date checked; re-read the current license before generating data.

## 1. DeepSeek-R1 (official GitHub README, section "7. License")
Source: https://github.com/deepseek-ai/DeepSeek-R1 README (fetched 2026-09-14; card [[deepseek-r1-distill-synth]] points to the same repository).

> This code repository and the model weights are licensed under the MIT License.
> DeepSeek-R1 series support commercial use, allow for any modifications and derivative works, including,
> but not limited to, distillation for training other LLMs. Please note that:
> - DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B and
>   DeepSeek-R1-Distill-Qwen-32B are derived from Qwen-2.5 series, which are originally licensed under
>   Apache 2.0 License, and now finetuned with 800k samples curated with DeepSeek-R1.
> - DeepSeek-R1-Distill-Llama-8B is derived from Llama3.1-8B-Base and is originally licensed under Llama3.1 license.
> - DeepSeek-R1-Distill-Llama-70B is derived from Llama3.3-70B-Instruct and is originally licensed under Llama3.3 license.

Note: the README's "Llama3.1-8B-Base" and the paper's Table 6 "Llama-3.1-8B" name the same base checkpoint.

## 2. Llama 3.1 (official Hugging Face model card, "Intended Use")
Source: https://huggingface.co/meta-llama/Llama-3.1-8B model card (fetched 2026-09-14).

> **License:** A custom commercial license, the Llama 3.1 Community License, is available at:
> https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE

> The Llama 3.1 model collection also supports the ability to leverage the outputs of its models to improve
> other models including synthetic data generation and distillation. The Llama 3.1 Community License allows
> for these use cases.

Not checked here: the full license text, its attribution and user-count clauses, and the Llama 3 (3.0)
license. Read the license file itself for those terms.

## 3. QwQ-32B-Preview and QwQ-32B (official Hugging Face model cards, metadata)
Sources: https://huggingface.co/Qwen/QwQ-32B-Preview and https://huggingface.co/Qwen/QwQ-32B (fetched 2026-09-14).

```
license: apache-2.0
license_link: https://huggingface.co/Qwen/QwQ-32B-Preview/blob/main/LICENSE
base_model: Qwen/Qwen2.5-32B-Instruct
```

The QwQ-32B-Preview card also lists known limitations, including "Language Mixing and Code-Switching" and
"Recursive Reasoning Loops".

## 4. Qwen2.5 (technical report, Table 1)
Source card: [[qwen-2.5]] — "Licenses: Apache 2.0 except 3B (Qwen Research) and 72B (Qwen) (Table 1)."

## 5. Closed-model terms of service
Source card: [[anthropic-distillation-attacks-report]] — Anthropic describes distillation as "a widely used and
legitimate method" when labs distill their own models, and reports campaigns that used fraudulent accounts
and proxy services "in violation of Anthropic's terms of service". The course library holds no verified
copy of other API providers' terms; check them directly.

## 6. Dataset licenses of the open reproductions (cards)
- Bespoke-Stratos-17k: Apache-2.0 ([[bespoke-stratos]], dataset card).
- OpenR1-Math-220k: Apache 2.0 ([[openr1]], Update #2 and dataset card).
- Dolphin: Apache-2.0 on the dataset; each model "follows the license of its base model" ([[dolphin]], card).
