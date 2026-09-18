<!-- scope: LDJnr/Capybara multi-turn synthetic SFT dataset (Amplify-Instruct) and the Nous-Capybara V1.9 model cards that describe its use
     deps: none
     see-also: [[airoboros]], [[baize]], [[ultrachat-construction]], [[openhermes]], [[dolphin]]
-->

# LDJnr/Capybara (Hugging Face dataset card)
- **Core Insight:** Capybara extends single-turn seed instructions from existing datasets into multi-turn conversations with a method named Amplify-Instruct whose details are unpublished; the released file has 16,006 conversations, of which 10,681 (66.7%) contain two or more user–assistant exchanges (derived from the released JSONL; see Technical Details).
- **Guideline:** When a synthetic SFT set is seeded from public instruction datasets, run a near-duplicate check against the evaluation benchmarks before release, because Capybara's MinHash check found MT-Bench overlap that had to be removed (dataset card, "Dataset contamination").
- **Authors:** Luigi Daniele (LDJ) and Suphavadeeprasit, as listed in the card's citation block; Nous Research (dataset-formation contributions credited to J-Supha)
- **Year:** 2023 (citation year; 7B model repo created 2023-10-10; dataset repo created 2023-12-16; the Amplify-Instruct paper is cited as "coming soon" with no identifier)
- **URL:** https://huggingface.co/datasets/LDJnr/Capybara (companions: https://huggingface.co/NousResearch/Nous-Capybara-7B-V1.9, https://huggingface.co/NousResearch/Nous-Capybara-34B)
- **Source type:** model/dataset card
- **Relevant topics:** multi-turn SFT data, synthetic dialogue, seed-based generation, benchmark decontamination, refusal-phrase filtering

## Summary
The dataset card presents Capybara as the first dataset built with Amplify-Instruct, a method that grows "thousands of high-quality single-turn seeds into advanced and in-depth multi-turn conversations" (dataset card, intro bullets). The card states that most tokens are newly synthesized, that conversations average over 1,000 tokens, and that conversations containing phrases associated with AI moralizing were removed. Neither the dataset card nor the model cards name the model that generated the turns, publish the generation prompts, or state how the number of turns was chosen. The model cards state that Nous-Capybara-7B V1.9 was trained for "multiple epochs" on roughly 20,000 examples and that Nous-Capybara-34B V1.9 was trained for 3 epochs; both cards list benchmarks as "Coming soon".

## Key Contributions
- A multi-turn SFT file, `CapybaraPure_Decontaminated.jsonl` (73,992,587 bytes, 16,006 rows, 13 source labels), under Apache-2.0 (HF file tree; Data Studio statistics; card metadata).
- A named generation method (Amplify-Instruct) that extends single-turn seeds into multi-turn conversations (dataset card intro). The cards describe the method only at this level.
- A phrase filter that removes any conversation containing "even a single instance" of overt AI moralizing, for example "As an AI language model", "September 2021", "I don't have personal beliefs" (dataset card, "Quality filtering and cleaning").
- A MinHash decontamination check at 100%, 99%, 98%, and 97% similarity against HumanEval, AGIEval, TruthfulQA, MMLU, and GPT4All, with MT-Bench overlap "cleaned out as of 12/15/2023" (dataset card, "Dataset contamination").
- SFT models trained on the data: Nous-Capybara-7B V1.9 and Nous-Capybara-34B V1.9 (model cards).

## Key Figures/Tables to Study
- Seed-source image (dataset card, "Thank you..." section, i.imgur.com/yB58OoD.jpeg): green marks datasets sampled for seeds; blue marks in-house curations that existed before Capybara. The list exists only as an image and is not transcribed here.
- Benchmark images (dataset card, "Benchmarks", i.imgur.com/OpajtNJ.jpeg and daIZn6n.jpeg): Capybara V1 on Llama-2 against other Llama-2 fine-tunes on AGIEval, BigBench, GPT4All, and the HF leaderboard. The numbers are in the images only and are not transcribed here.

## Technical Details
**Seeds (as stated by the cards).**
- The dataset card names Evol-Instruct, Alpaca, Orca, Vicuna, Lamini, and FLASK as synthesis techniques whose insights Capybara builds on (dataset card intro). The 7B card describes the "seed distribution and synthesis method" as a combination of Airoboros, Evol-Instruct (WizardLM), Orca, Vicuna, Know_Logic, Lamini, FLASK "and others" (7B card, paragraph 3).
- Seed datasets named: Airoboros, Know Logic, EverythingLM, GPTeacher, new seed instructions from other sources, and in-house multi-turn datasets Dove and Verified-Camel ("A successor to Puffin") (dataset card intro). The 7B card adds seed instructions "derived from posts on the website LessWrong" (7B card, paragraph 3).
- "Additional data came from human curated CamelAI data, with the help of volunteers ranging from former Physics PhD's, Mathematicians, Biologists and more" (7B card, "Model Training"). Removing "mathematically/verifiably incorrect answers" with expert volunteers is listed as a future plan (dataset card, "Future Plans").

**Size and length claims.** "Over 10,000 multi-turn examples" (dataset card heading). The fine-tuning set is "entirely contained within 20K training examples", described as "10 times smaller than many similar performing datasets" without naming those datasets (dataset card intro). "Over 60%" of the data is multi-turn and "Over 1,000 tokens average per conversation example" (7B card, "Notable Features"). The dataset card also says "3 turns or more per example" (intro bullets).

**Released-file statistics (derived).** Computed on 2026-09-14 from `CapybaraPure_Decontaminated.jsonl` at dataset commit c2bc39a. One exchange = one `input`/`output` pair in the `conversation` list.
- Exchanges per row: 1 → 5,325 rows; 2 → 753; 3 → 6,008; 4 → 3,656; 5–17 → 264. Mean 2.58, median 3, maximum 17; these match the Data Studio statistics (mean 2.58034, median 3.0).
- Rows with ≥2 exchanges: 10,681 / 16,006 = 66.7%, consistent with "over 60%". The "3 turns or more per example" statement does not hold for the 5,325 single-exchange rows.
- Whitespace-delimited words per row, all turns: mean 695, median 626. Token counts were not computed because they depend on the tokenizer.
- Rows by source label (share with ≥2 exchanges): Dove 3,703 (32.6%); Airoboros 3,516 (100%); Know-Logic 1,867 (99.9%); EverythingLM 1,344 (100%); GOAT 1,000 (0%); TaskSource 889 (0%); General-Instruct 846 (100%); GPT4LLM 820 (100%); TheoremQA 795 (0%); Less-Wrong 656 (100%); SuperCOT 428 (100%); Verified-Camel 127 (0%); Tigerbot 15 (0%).
- Follow-up phrasing: in the 10,681 rows with ≥2 exchanges, the second user turn starts with "Considering" in 1,787 rows and "Given" in 1,090. In the 9,928 rows with ≥3 exchanges, the third user turn starts with "Reflecting" in 1,275 rows and "Considering" in 1,226. In rows inspected by hand, such openings refer to the previous answer (for example "Considering the vast number of islands in Sweden, how has this geographical feature influenced...", a General-Instruct row in the dataset viewer). The cards report no test of whether later turns depend on earlier ones.

**Model cards.** The 7B card metadata lists license `mit` and tag `StableLM`, while its config.json has `_name_or_path: mistralai/Mistral-7B-v0.1` and `model_type: mistral` (7B card YAML; config.json). The 7B card states that V1.9 uses "novel unalignment techniques" without describing them ("What's new compared to V1?").

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Nous-Capybara-7B V1.9 | 7B | SFT | base checkpoint | `mistralai/Mistral-7B-v0.1` | HF NousResearch/Nous-Capybara-7B-V1.9@ea08e10 config.json | verified 2026-09-14 | no ablation reported |
| Nous-Capybara-7B V1.9 | 7B | SFT | number of examples | "roughly 20,000 carefully curated conversational examples" | same repo, README "Model Training" | verified 2026-09-14 | no ablation reported |
| Nous-Capybara-7B V1.9 | 7B | SFT | epochs | "multiple epochs" | same repo, README "Model Training" | verified 2026-09-14 | no ablation reported |
| Nous-Capybara-7B V1.9 | 7B | SFT | datasets listed | LDJnr/Capybara, LDJnr/LessWrong-Amplify-Instruct, LDJnr/Pure-Dove, LDJnr/Verified-Camel | same repo, README YAML `datasets` | verified 2026-09-14 | no ablation reported |
| Nous-Capybara-7B V1.9 | 7B | SFT | prompt format | `USER:` / `ASSISTANT:` | same repo, README "Prompt Format" | verified 2026-09-14 | no ablation reported |
| Nous-Capybara-34B V1.9 | 34B | SFT | base checkpoint; epochs | Yi-34B with 200K context length; 3 epochs on the Capybara dataset | HF NousResearch/Nous-Capybara-34B@6beb706 README heading | verified 2026-09-14 | no ablation reported |
| Nous-Capybara-34B V1.9 | 34B | SFT | prompt format | prefix `USER:`, suffix `ASSISTANT:`, stop token `</s>` | same repo, README "Prompt Format" | verified 2026-09-14 | no ablation reported |
| Nous-Capybara-7B / 34B V1.9 | 7B, 34B | SFT | LR, batch, sequence length, loss masking, packing, compute | not reported | checked: both model READMEs, 7B config.json, dataset card | not reported | — |

## Findings relevant to generality and long context
- Generality: the dataset card claims "a strong focus on information diversity across a wide range of domains" but reports no diversity measurement (dataset card intro). Its benchmark images refer to Capybara V1 on Llama-2, not to V1.9 (dataset card, "Benchmarks").
- Contamination measurement: MinHash similarity on benchmark questions and answers at 97–100% thresholds; MT-Bench overlap was found and removed; the other five benchmarks showed no match down to 97% (dataset card, "Dataset contamination").
- Long context: conversations average over 1,000 tokens (7B card). The 34B model uses a 200K-context base, but its card reports no long-context evaluation and no training sequence length (34B card).

## Connections
- [[airoboros]] — named seed dataset; 3,516 released rows carry the Airoboros label.
- [[baize]] — multi-turn data by self-chat from seed topics; Capybara extends seed instructions taken from existing datasets.
- [[ultrachat-construction]] — multi-turn synthesis with a separate user-simulator model and documented user-turn generation, which the Capybara cards do not document.
- [[openhermes]] — another community SFT mixture assembled from several synthetic sources.
- [[dolphin]] — the dataset page lists dolphin-2.6-mixtral-8x7b among models fine-tuned on LDJnr/Capybara.

## Verification
- Checked on 2026-09-14 against: https://huggingface.co/datasets/LDJnr/Capybara (README and JSONL at commit c2bc39a; Data Studio statistics); https://huggingface.co/NousResearch/Nous-Capybara-7B-V1.9 (commit ea08e10; README, config.json); https://huggingface.co/NousResearch/Nous-Capybara-34B (commit 6beb706; README).
- Corrections to the previous card version:
  - "~20K carefully curated conversations" → the released file has 16,006 rows; "roughly 20,000" describes the 7B V1.9 training set (7B card, "Model Training").
  - Evol-Instruct, Orca, Vicuna, Lamini, FLASK listed as seed sources → the dataset card names them as synthesis techniques; named seed datasets are Airoboros, Know Logic, EverythingLM, GPTeacher, LessWrong posts, Dove, and Verified-Camel.
  - EverythingLM and GPTeacher listed as "curated human posts" → the cards list them with Airoboros as seed datasets.
  - "volunteer-curated physics / math / biology / chemistry prompts via CamelAI partnership" → "human curated CamelAI data, with the help of volunteers"; no partnership is stated.
  - "Dove dataset extensions" → Dove is an in-house multi-turn dataset used as seeds.
  - Authors "LDJnr (Daniele), J-Supha, Luigi, Suphavadeeprasit" → citation lists "Daniele, Luigi and Suphavadeeprasit"; LDJ is "Luigi D."; J-Supha is credited for dataset formation.
  - "model: Mistral base license" → the 7B card metadata license is `mit`.
  - "Year: 2023–2024" → 2023.
  - "Exact prompt templates documented on the model card" → the model cards document only the inference prompt format.
  - "Nous-Capybara-34B (Yi-34B base) pushed it further" → the 34B card confirms the Yi-34B-200K base and 3 epochs but reports no benchmark results.
- Removed as unsupported by the source: the six-step Amplify-Instruct schematic (distributional samplers, "2–6" turns, length/dedup/LLM-check filtering); "small enough for LoRA on a single consumer GPU"; "strong MT-Bench / AGIEval numbers at release" for V1.9; "Evaluation emphasizes multi-turn coherence and reasoning probes"; the practitioner takeaways ("Seed diversity > raw scale", "compact size is deliberate", "under-explored"); "Volunteer-labeled subsets have uneven quality control"; "outputs from GPT-4 era can carry API terms"; "Seed composition idea influenced later community mixes (OpenHermes 2.5, Dolphin)".
- Not reported by the source: the generator model(s); synthesis prompts; the rule for the number of turns; any check that follow-ups depend on earlier turns; per-tokenizer token counts; training hyperparameters; benchmark numbers for V1.9 and 34B.
