---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: "OLMES: A Standard for Language Model Evaluations (Gu, Tafjord, Kuehl, Haddad, Dodge, Hajishirzi)"
source_url: https://arxiv.org/abs/2406.08446
created_at: "2026-09-17"
note: "No library card exists for the slug `olmes` in wiki/raw-data/llm-training/ as of 2026-09-17. Every number below was read in the cached full text of arXiv:2406.08446v2."
---

# Excerpt: OLMES — a standard for in-loop and published LM evaluation

- **Authors:** Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge, Hannaneh Hajishirzi (Allen Institute for AI; University of Washington)
- **Year:** arXiv v1 2024-06; read at v2 (2025-02-11)
- **Source type:** paper. Code and prompts: https://github.com/allenai/olmes

## The problem the standard addresses (§1, Table 1)
Published accuracies for the same model on the same dataset differ by evaluation setup. Table 1 lists
ARC-Challenge scores collected from six references:

| Model | range of reported scores | setups that differ |
|---|---|---|
| Llama2-7B | 43.2 – 53.7 | 0-shot vs 25-shot; CF vs MCF; char / pmi / none normalization |
| Llama2-13B | 48.8 – 67.6 | same |
| Llama3-8B | 60.2 (25-shot CF, HF Open LLM Leaderboard) vs 78.6 (25-shot MCF, Llama 3 model card) | formulation |

The paper cites Sclar et al. (2023) for accuracy differences of as much as 80% on one task from formatting
and in-context example changes alone (§1).

## Definitions (§2.1, §3.3)
- **MCF** (multiple-choice formulation): the answer options appear in the prompt with letter labels; the model
  is scored on the label token.
- **CF** (cloze / completion formulation): each answer string is scored separately by its own probability.
- CF normalizations: `none` = ln P(a|q); `token` = ln P(a|q) / number of tokens in a; `character` =
  ln P(a|q) / number of characters in a; `pmi` = ln [ P(a|q) / P(a|u) ] with the uninformative prompt u = "Answer:".

## Findings used by ch-06
- **Format acquisition during training (Fig. 1).** OLMo-7B-0424 evaluated on the MMLU validation set in both
  formats: MCF sits near random until roughly 400B training tokens, then becomes the stronger signal, while CF
  gives signal from early training and levels off late (§3.4, Fig. 1).
- **Across 15 base models on ARC-Challenge (Fig. 2):** the weakest 8 models are near random under MCF but above
  random under CF; strong models score higher under MCF.
- **Normalization choice matters less than formulation:** the OLMES per-task normalization is within 0.0–1.1
  points of the per-model oracle normalization (§3.3, Table 3).

## The standard (§3.1–3.5, §4)
- Test split when labels are public, otherwise validation; sample 1,000 instances when the split has more
  than 1,500, using `Random(1234).sample(all_instances, 1000)`.
- Exact prompt format specified per task; fixed, manually curated 5-shot examples.
- Prescribed CF normalization per task; `pmi` for ARC-Challenge, CommonsenseQA, OpenBookQA; `character` for
  ARC-Easy, HellaSwag, PIQA, Social IQa, MMLU; `none` for BoolQ, WinoGrande.
- Evaluate with both MCF and CF and report the better of the two (§3.4).
- MMLU uses the macro average over 57 subjects; inputs capped at 2,048 tokens; default model precision;
  two newlines between in-context examples.

## Limits stated by the source
- The standard covers multiple-choice tasks for base models. Generative tasks, chain-of-thought prompting, and
  chat-model message formatting are named as future work (Limitations).

## Verification
- Read on 2026-09-17 in the cached full text of arXiv:2406.08446v2 (§1–§4, Tables 1–3, Figs. 1–2, Limitations).
