<!-- scope: HuggingFaceTB/smoltalk dataset card — the 1M-sample SFT mix used to build SmolLM2-Instruct, combining four new synthetic sets with subsets of eight public datasets
     deps: [[magpie]]
     see-also: [[hf-cosmopedia]], [[tulu-3-sft-mix]], [[openhermes]], [[smollm-3]]
-->

# SmolTalk (HuggingFaceTB/smoltalk dataset card)
- **Core Insight:** SmolTalk is a 1M-sample synthetic SFT mix whose core component is Smol-Magpie-Ultra, 400K samples generated with the Magpie pipeline using Llama-3.1-405B-Instruct; the card states that SmolLM models trained on Smol-Magpie-Ultra alone outperform models trained on OpenHermes and Magpie Pro on benchmarks including IFEval and MT-Bench (Dataset composition).
- **Guideline:** When a small-model SFT mix is assembled from public data only, add targeted synthetic sets for the capabilities the general mix underweights and add public subsets chosen by ablation, because the authors report that models finetuned on public SFT datasets underperformed models trained on proprietary instruction data, and they selected each public subset by finetuning SmolLM2-1.7B on candidates and keeping tuned ratios of the best performing ones (Dataset description; Existing public datasets).
- **Authors:** Hugging Face SmolLM team; the card's citation is the SmolLM2 paper by Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. (Citation)
- **Year:** 2025 for the cited paper (arXiv:2502.02737, v1 2025-02); the dataset card itself prints no release date
- **URL:** https://huggingface.co/datasets/HuggingFaceTB/smoltalk (paper: https://arxiv.org/abs/2502.02737 ; generation code: https://github.com/huggingface/smollm/tree/main/text/data/smoltalk)
- **Source type:** model/dataset card
- **Relevant topics:** SFT mixture composition, Magpie-style synthesis, distilabel, small-model post-training, long-context SFT data

## Summary
SmolTalk is a synthetic dataset for supervised fine-tuning, used to build the SmolLM2-Instruct family, containing 1M samples (Dataset description). The card states the motivation: during SmolLM2 development, models finetuned on public SFT datasets underperformed models with proprietary instruction data, so the team created new synthetic datasets covering text editing, rewriting, summarization, and reasoning, then ran data ablations at 1.7B scale to add public datasets for mathematics, coding, system-prompt following, and long-context understanding (Dataset description). All new datasets were generated with distilabel (Dataset description).

## Key Contributions
- Four new synthetic subsets released under Apache 2.0: Smol-Magpie-Ultra, Smol-constraints, Smol-rewrite, Smol-summarize (Dataset composition; License).
- Ablation-selected subsets of public datasets rather than whole datasets, with tuned ratios (Existing public datasets).
- Decontamination of Smol-constraints against IFEval (Dataset composition).
- Smol-SmolTalk, a reduced variant for SmolLM2-135M-Instruct and SmolLM2-360M-Instruct that keeps only more concise Smol-Magpie-Ultra conversations and excludes advanced math (Smol-SmolTalk).
- A comparison against Orca AgentInstruct 1M under a fixed training setup (Evaluation).

## Key Figures/Tables to Study
- The dataset-viewer subset list (dataset page header) — per-subset row counts, which differ from the prose sample counts.
- The Evaluation section — SmolTalk versus AgentInstruct 1M at SmolLM2 scale, and the Mistral-7B result.

## Technical Details
Prose sample counts and dataset-viewer row counts, both from the dataset page (Dataset composition; viewer subset list):

| Subset | Count stated in prose | Rows in viewer | Purpose stated by the card |
|---|---|---|---|
| smol-magpie-ultra | 400K | 431k | Magpie pipeline with Llama-3.1-405B-Instruct; curated and filtered more heavily than Magpie-Pro |
| smol-constraints | 36K | 36.2k | constraint following (fixed sentence/word counts, required words); decontaminated against IFEval |
| smol-rewrite | 50k | 56.2k | tone and style rewriting |
| smol-summarize | 100k | 101k | email and news summarization |
| openhermes-100k | 100k | 100k | preserves and boosts MMLU, WinoGrande, BBH |
| metamathqa-50k | 50k random samples | 50k | mathematics and reasoning |
| numina-cot-100k | not stated | 112k | hard math, benchmarks such as MATH |
| self-oss-instruct | not stated | 50.7k | coding |
| systemchats-30k | 30k | 35.9k | varied system-prompt formats |
| longalign | not stated | 3.73k | English LongAlign-10k samples under 16k tokens |
| everyday-conversations | not stated | 2.38k | multi-turn everyday dialogue, from SmolLM v1 post-training |
| apigen-80k | 80k | 87.5k | function calling; mix of Synth-APIGen-v0.1 and xlam-function-calling-60k |
| explore-instruct-rewriting | 30k | 32k | instruction rewriting |
| all | 1M | 1.1M | full mix |

- Evaluation setup for the SmolTalk vs Orca AgentInstruct 1M comparison: SmolLM2 finetuned on each dataset with the same setup — 2 epochs, learning rate 3e-04, sequence length 8192, global batch size 16 (Evaluation). The card does not print the resulting scores in text.
- At 7B scale, the card reports significant improvements when finetuning Mistral-7B on SmolTalk, named on IFEval, BBH, GSM8K, and MATH, without numbers (Evaluation).
- Licensing: the four new datasets are Apache 2.0; public subsets retain their original licenses (License).
- Training code for SmolLM2 is the alignment-handbook SmolLM2 recipe (Smol-SmolTalk).

## Findings relevant to generality and long context
- Long context: the card states that finetuning only on short samples makes the model lose long-context ability beyond 2048 tokens, which is why English LongAlign-10k samples under 16k tokens were added and training used a sequence length of 8192 (Existing public datasets). No long-context benchmark result is reported for this choice, and the LongAlign slice is 3.73k rows of the 1.1M-row mix.
- Generality across capabilities: each public subset is attributed to a capability the team measured at 1.7B scale (math, coding, system prompts, long context), but the card reports no per-subset ablation numbers (Existing public datasets).
- Multi-turn and system-prompt data: SystemChats2.0 supplies system-prompt variety, and the card notes Smol-rewrite and Smol-summarize also contain system prompts (Existing public datasets). Conversation lengths and turn counts are not reported.
- Size-matched mixes: smaller SmolLM2 variants use Smol-SmolTalk rather than the full mix, which the card justifies by conversation length and math difficulty rather than by a measured score (Smol-SmolTalk).

## Connections
- [[magpie]] — the generation pipeline behind Smol-Magpie-Ultra.
- [[openhermes]] — the 100k subset included here.
- [[tulu-3-sft-mix]] — a comparable open SFT mixture with different components.
- [[hf-cosmopedia]] — earlier Hugging Face synthetic corpus, on the pretraining side.
- [[smollm-3]] — the successor model whose SFT mixture (SmolTalk2) replaces this one.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/datasets/HuggingFaceTB/smoltalk (dataset card and viewer subset list).
- Corrections to the previous card version:
  - Missing component added: APIGen-Function-Calling, 80k samples from a mix of Synth-APIGen-v0.1 and xlam-function-calling-60k (Existing public datasets).
  - "Explore-Instruct-Rewriting | (subset)" → 30k samples (Existing public datasets).
  - "SystemChats 2.0 | 30K" kept, with the viewer's 35.9k rows recorded alongside it, since the two counts differ.
  - "Smol-rewrite 50K" and "Smol-summarize 100K" kept as printed in prose; viewer rows are 56.2k and 101k.
  - "SmolLM2-1.7B-Instruct ... is best-in-class small model at release on IFEval, MMLU-Pro, and BBH for its size" → the dataset card makes no such claim; it reports a comparison against Orca AgentInstruct 1M under a fixed setup and unquantified gains for Mistral-7B on IFEval, BBH, GSM8K, MATH (Evaluation).
  - "LongAlign: long-context" → the card states the specific reason (loss of long-context ability beyond 2048 tokens), the filter (English, under 16k tokens), and the training sequence length (8192).
  - Author line replaced by the card's own citation, which is the SmolLM2 paper author list.
- Removed as unsupported by the source: "Magpie-Ultra (40%) ... cheap (no API) because Llama-3.1-405B is open"; "Open Magpie-style synthesis at ~400K scale is the new default open SFT anchor"; "Composition table is tunable — for a coding-focused variant, inflate the code share"; "Teacher-model inheritance: Llama-3.1-405B-Instruct biases propagate"; "OpenHermes 2.5 already contains Magpie-adjacent data — dedup matters"; "downstream long-context performance partly inherited from pretraining"; "Explicit targeted synthesis ... is more effective than assuming Magpie covers them"; the claim that this recipe "validates Magpie-scale synthetic SFT in the open" as a measured result.
- Not reported by the source: the numeric scores behind the AgentInstruct comparison and the Mistral-7B result, per-subset ablation deltas, token counts (rows only), deduplication procedure across subsets, generation cost, and the teacher models used for subsets other than Smol-Magpie-Ultra.
