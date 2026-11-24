# 23a — What Knowledge-Tail Papers Actually Show

<!-- scope: primary-source setups, measured outcomes, and transfer limits for rarity units, sampling support, retention bias, and source-grounded knowledge coverage
     deps: [[read]]
     see-also: [[read]], [[practice]], [[long-tail-papers]]
     sources: [[long-tail-knowledge]], [[graphgen]], [[knowledge-density-synthesis]], [[data-value-scaling]], [[long-tail-knowledge-survey]]
-->

## 1. Relevant documents and factual QA

**Result (single study).** Kandpal et al., [[long-tail-knowledge]], link salient question and answer entities to pretraining corpora and count documents containing both (§2). They evaluate TriviaQA and Natural Questions across frequency bins. For BLOOM-176B, TriviaQA accuracy rises from about 25% to above 55% as relevant-document counts rise from tens to ten-thousands (§3.1). This is an association across questions, not a controlled replacement of conversation styles.

The paper also retrains a 4.8B model after removing relevant documents from C4, probing whether exposure matters (§3.2). Retrieval improves performance especially on low-frequency questions (§4.3). The evidence supports supplying missing factual material, not assuming all rare dialogue failures are missing knowledge. Entity counts depend on entity linking and the chosen corpora; a document containing two entities is only a proxy for useful evidence.

## 2. GraphGen: structure before synthesis

**Result (single study).** [[graphgen]] constructs a knowledge graph from source text, assesses the model's comprehension, and generates questions at atomic, aggregated, and multi-hop levels. Atomic questions isolate one relation; aggregation brings several relevant relations together; multi-hop questions require traversing linked facts. This changes which facts and combinations get training exposure, rather than merely rephrasing frequent facts.

In §4 Step 2, Eq. 4 uses cross-entropy over true and negated statements for comprehension assessment. The abstract describes the broader assessment using ECE terminology. Do not replace this with the claim that the method is simply ordinary binned calibration error. A low-confidence or high-loss statement is an estimated gap under that probe, not a proven important real-world tail.

**Interpretation.** For product facts, an analogous pipeline could extract supported relations, probe them, and create grounded questions. The graph and negations must be verified: a faulty relation or an invalid negation contaminates the target. For appointment corrections or skill scope, a factual graph is not sufficient unless the relevant procedural conditions are represented. Transfer to the sales setting remains an **Open question**.

## 3. KDoS and mixed-data scaling

**Result (single study).** [[knowledge-density-synthesis]] proposes feedback from knowledge density to adjust synthesis distribution. Its demonstrated setting is post-training/SFT; the Limitations section does not establish pretraining gains or arbitrary conversational skill transfer. The useful mechanism is a feedback loop: characterize under-covered knowledge, allocate synthesis toward it, and remeasure. It is not a source for a universally optimal rare-data share.

[[data-value-scaling]] analyzes different head/tail learning phases with mixed real and synthetic data and proposes data valuation. Its §5.2 three-phase demonstration is a controlled ResNet-18/CIFAR-100 experiment, with corrupted CIFAR-100-C treated as synthetic, Zipf-weighted classes, and a real fraction of 0.0625. That is not natural LLM conversation-tail evidence; NLP subset-utility experiments are a separate result. [[long-tail-knowledge-survey]] is a **Review**, organizing definitions and interventions; it supplies perspective rather than another controlled demonstration.

## 4. Study setup and measured outcomes

| Study | Setup and metric | Result and boundary |
|---|---|---|
| GraphGen §5/Fig.3 | Qwen2.5-72B synthesis, 7B-Instruct student; agriculture, medical, multi-hop QA | ROUGE-F gains of 1.08/2.70/4.73 points over best baselines; not factual-accuracy gains |
| KDoS Table 1 | Qwen3-4B-Base, benchmark average | Random 18.3, Uniform 20.9, Quality 20.6, Difficulty 21.5, KDoS 22.8; Difficulty beats KDoS on SimpleQA-Verified, 8.2 versus 8.0 |

KDoS density is token count divided by semantic-hypersphere volume, with radius estimated from mean embedding distance to the centroid (§3.1). It is not factual relations per token or an intrinsic universal quality measure. Groups and two nearest neighbors condition five candidate questions. Quality checking and density-based rejection are separate; cold start accepts quality candidates before density control. Table 2 reports full 22.8 versus 21.7 without quality and 17.6 without quality/density; the seed is 17.4. These are source-specific scores, not a sales prescription.

GraphGen Table 2 also changes average example length: 657.9 tokens versus 357.1 for LongForm. Therefore a comparison of lexical variety or downstream utility must consider length and training exposure; the method label alone does not isolate their contribution.

## 5. Evidence boundaries

A result on closed-book QA can motivate a hypothesis about factual coverage in conversation, but does not demonstrate tone realism, speaker coverage, or a correct tool call. A gap score can prioritize investigation, but the probe itself has blind spots. Use independent evaluation items that are not merely the synthesis questions with new names. Keep the strongest claim no broader than the source setting.

[Back to lesson](read.md) · [Implementation and evaluation](practice.md) · [한국어](methods_kor.md).

