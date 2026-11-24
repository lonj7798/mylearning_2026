# DeepDialogue: A Multi-Turn Emotionally-Rich Spoken Dialogue Dataset

<!-- scope: DeepDialogue; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** Domain/emotion scaffolds support emotionally varied synthetic text and speech.
- **Guideline:** Audit which emotional and length slices survive filtering.
- **Authors:** Alkis Koudounas, Moreno La Quatra, Elena Baralis
- **Year:** 2025
- **URL:** [Primary source](https://arxiv.org/abs/2505.19978)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

Generation uses domain and emotion stratification with an emotion-transition graph; retained text is converted to expressive speech.

## Key Contributions

- Result (single study): Domain/emotion scaffolds support emotionally varied synthetic text and speech.
- Interpretation: Audit which emotional and length slices survive filtering.

## Key Figures/Tables to Study

Figures/tables not extracted. The Technical Details section identifies verified method sections where available; numerical outcome comparisons require the primary text.

## Technical Details

§§3.1–3.2: history/domain/emotion conditioning. App.B.2: longer dialogues fail QC more; App.B.3 maps twenty text emotion labels to eight audio categories.

## Limits

TTS does not demonstrate natural overlap or accent coverage. §4.2 evaluates speech emotion recognition, not sales behavior.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Primary full text and appendices verified on 2026-10-06; first release 2025-05-26.

## Full-text follow-up (2026-10-07)

§§3.1–3.2: 41 domains, 20 text emotions, nine generation models; 65,600 initial candidates and 40,150 retained dialogues; maximum 25 words per response in construction. App. B.2/B.3 and §4.2 establish length-filter bias, eight mapped audio classes, and speech-emotion-recognition rather than sales evaluation. Verified full text on 2026-10-07.
