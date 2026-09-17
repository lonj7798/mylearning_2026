---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "The Leaderboard Illusion (arXiv:2504.20879v2, 2025-05-12; v1 2025-04-29)"
source_url: https://arxiv.org/abs/2504.20879
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug leaderboard-illusion). Values read from the v2 PDF on 2026-09-15."
---

# Excerpt: selective reporting and data asymmetry on Chatbot Arena

**Authors:** Shivalika Singh, Yiyang Nan, Alex Wang, Daniel D'souza, Sayash Kapoor, Ahmet Üstün, Sanmi Koyejo, Yuntian Deng, Shayne Longpre, Noah A. Smith, Beyza Ermis, Marzieh Fadaee, Sara Hooker (Cohere Labs and co-authors at Princeton, Stanford, Waterloo, MIT, AI2, UW). Several authors have submitted models to the Arena, which the paper states explicitly.

## Private testing and selective disclosure (§3)

- Providers may test multiple private variants before public release and retract scores. At the extreme, Meta tested 27 private variants before the Llama-4 launch (§3.1).
- Simulation: as the number of privately tested variants rises from 0 to 50, the expected maximum discovered Arena Score rises; testing 20 variants yields about +50 points over a single submission (§3.2, Fig. 7).
- Real-world control: the authors submitted **two identical checkpoints** of Aya-Vision-8B in March 2025 without disclosing that they were identical. Final Arena scores were 1052 (±21/22) and 1069 (±19/23), with 4 other models ranked between them (§3.3, Fig. 9).
- Two genuinely different Aya-Vision-32B variants scored 1097 (±29/25) and 1059 (±18/23), with 9 models between them (§3.3).

## Data access asymmetry (§4.1)

- Estimated share of all Arena data: OpenAI 20.4%, Google 19.2%; 83 open-weight models together 29.7%; 41 fully open-source models together 8.9%; OpenAI, Google, Meta and Anthropic together 62.8%.
- Maximum daily sampling rate by provider (January–March 2025 scrape): Google and OpenAI 34%, xAI 22.0%, Meta 17.9%, Reka 3.3%.
- 205 models were silently deprecated against 47 listed as deprecated in the FastChat backend; 64% of the silently deprecated models are open-weight or open-source (§5).

## The overfitting experiment (§4.2, Fig. 10, App. Table 9)

- A 7B base model used for the Cohere Command family is fine-tuned three times with identical settings, varying only the share of *arena-mix* data (samples from Arena battles) against *other-sft-mix* (a proprietary instruction, multilingual, math and code mixture): 0%, 30%, 70%. All three run 1.3K steps at batch size 128. No hyperparameter sweep was done.
- In-distribution measurement: win rate on the 500 English ArenaHard prompts, judged by gpt-4o-2024-11-20, against Llama-3.1-8B-Instruct.

| Arena share | ArenaHard win rate vs Llama-3.1-8B-Instruct | MMLU |
|---|---|---|
| 0% | 23.5% | 66.5% |
| 30% | 42.7% | 64.4% |
| 70% | 49.9% | 65.9% |

- Relative win-rate gains are 81.7% (30%) and 112.3% (70%) (§4.2). MMLU does not improve. The authors read this as gains specific to the Arena distribution rather than general capability.
- Basis for calling ArenaHard in-distribution: the authors cite its reported 98.6% correlation with Arena human preference rankings (§4.2).
- The authors also note that a provider need not train directly on the data: the composition can guide data weighting or synthetic-data generation (§4.2).

## Limits

- Data shares and sampling rates are estimates from a scrape covering a specific window (January–March 2025, plus historical battle data); the Arena does not publish per-provider totals.
- The fine-tuning experiment uses one 7B model, one mixture family, and one judge; the authors describe it as a lower bound and did not tune it.

## Used in

ch-47a §6 (leaderboard selective reporting and benchmark-specific training), §7, Recipe, Generalization lens.
