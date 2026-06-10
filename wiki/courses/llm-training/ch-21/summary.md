<!-- chapter: ch-21 — learner summary
     deps: [[read]], [[qa]]
     scope: learner-authored distillation; written alone, no teacher input on content
-->

# Ch-21 — Learner Summary

## Core insight (one sentence)

About Taxonomy-driven and textbook synthesis. 
Taxonomy-driven synthetis is a way to cover the diverse topics compared with 'seed-based' approaches. this method can cover lots of topics compared with seed-based approaches, but limited in the taxonomy, taxonomy-generation model, and less creativity(imaginary compared with human seed-data)

textbook synthetis is a way to improve the model from 'pretraining stage'. baisically synthesize the raw-data into a 'textbook-like' data using a model. this method can improve the model's performance from the area where the data is synthesized. and actually this may cause some bias and inequivalent performance across the topics (bias from the human curator who decide the scope of the taxonomy).

## What taxonomy-driven synthesis is — and what makes it different

taxonomy-driven synthesis is generate as many as topic instruction data. starting from the core taxonomy and expand it into the 'leaf' using a model (creating leaf can be multi-step process). 
and this is top-down method. 
and those edge(very last part, end of leaf) of the taxonomy can be used to generate the synthetic data like extracting the knowledge. 
BUT this method has a clear limitation. it cannot overcome the ceiling of the teacher model. because it generates the data using the teacher model itself and learn that data. 

## §1 GLAN — taxonomy as the whole design document

GLAN is a taxonomy-driven synthesis method, even without seed data. 
basically decompose human knowledge into a tree shape. 
and then generate instruction data per leaf. 


## §2 Nemotron-4 — taxonomy + RM-as-filter-and-judge

Nemotron-4 is also used taxonomy-driven synthesis data. but unlike GLAN, it used the seed data (6 different human curated tasks) to guide the taxonomy generation.
most impressive part of Nemotron-4 is that >98% of the data is synthesized. leverage RM as a both filter and judge. for filtering the data, scored which type of data they need to use to train the model. and for judger, ranked which kind of data is better and used those signal during SFT and DPO.

## §3 Phi — textbook synthesis at pretraining scale

Phi used 'textbook synthesis' method at pretraining stage. 
Phi shows quality matters more than the quantity.
Phi synthesize the data from topic list and adds synthetic textbook-like data generated from topic list. (this improve the performance of that specific topics)
and this kind of systematic curator bias affects the model to have a narrow capability especially on the area where the textbook-like data is not used. 

## §4 Cosmopedia — open replication and the dedup lesson

Cosmopedia has interesting insight. 
"prompts must be rewritten carefully to avoid near-duplicate generations"
they dedpup the dataset on top of the taxonomy, and the found those deduplication is comming from the generation prompt. so when we approach with taxonomy-driven synthesis, we need to carefuly curate the prompt to avoid those near-duplicate generations.
they rewrite the noisy web crawl data(seed) into a textbook-like data, and with the mixture of human data + synthetic data(textbook-like) they improved the small model's performance with less training tokens. 

## Bottom-up vs Top-down — the 3-layer architecture

Top-down provides wider coverage. each leaf of the tree can synthesize N samples, and this give equivalent coverage for each topic in the taxonomy. But the problem is 
1. if we don't focus on the generation prompt, most of the data will be very similar to each other. (as shown in cosmopedia)
2. top-down cannot overcome the ceiling of the teacher model (less creativity compared with Bottom-up)

Bottom-up provides narrow coverage compared with top-down, but still can provide human-level insight because bottom-up method uses human seed data, and it can break  prompt distribution ceiling of the teacher model. 

here is the 3 layer architecture: 
1. generate
2. dedup
3. verify. 
and if fullfill all of them from Bottom-up, it can be a high quality synthetic data. 

## Framework — 4-axis predictor (verifier / taxonomy / long-tail / substrate)

- verifier: can help the correctness of the data. so baisically how clean it is. 
- taxonomy: can help the diversity and coverage of the data. it can help to fill-out the gap of missing topics. 
- long-tail(rare-case): I believe this part is very important. if there are some instruction or query that model haven't seen during the training and then facing long-tail case during inference, the trained model will easily break. so this part will help model to be robust. (maybe really hard to cover with synthetic data.)
- language substrate(언어 기질): this part covers the 'does model understand the language?' part. this is quite critical during the training but also inference. if the model trained without any language substrate, then it may learn the style only, and cannot generalize the context (I would say understand). if there are unseen query, but if the model already has strong language substrate, then model can generalize that unseen query to the data what model has already seen before. also even from the math domain, without substrate, model cannot understand the question itself. 

if possible to improve the verifier and taxonomy coverage, then we can expect 'clean' and 'structured' data. 
if possible to improve the long-tail and substrate, we can expect 'real-web' like data. 
so it is important to find the right balance of these 4-axes. 


## Failure modes (cross-cutting)

- Top-down: cannot break the ceiling of the teacher model
- contamination
- curator blind spot
- cosmetic vs sctructural variation
- Bottom-up: lack of coverage. 

## Connections to other chapters

*Backward links (covered chapters)*:
- [[ch-18]] — top-down taxonomy is one specialization of the *Stage-1 full-generate* category in the generate→filter→dedup→verify loop. RM-as-judge (Nemotron) is the filter+verify specialization. Cosmopedia 40% dedup signal is the *upstream-restructure-not-downstream-filter* lesson applied to the ch-18 loop's dedup step.
- [[ch-19]] — top-down is the **direct answer to ch-19 verdict E5 diversity ceiling**: seed-bound composition (Bootstrap/Evol/Persona/Humpback) limited by what 175-500 seeds happen to span; top-down replaces seed pool with curator's tree → ceiling moves from *invisible seed-bound* to *visible curator-bound*. Bottom-up still wins on *cross-branch combinations* (line 219 white-space-between-branches).
- [[ch-20]] — ch-20's 9-axis framework directly maps the 3 papers: GLAN (axis 9 max + axis 5 = 0), Nemotron (axis 5 + axis 9 both strong via RM), Phi (axis 9 + style filter, no verifier). Staged SFT (code → general) is mentioned at line 152, the cleaner-signal-first principle from ch-20 R1 4-stage isomorphism.

*Forward links (unread — surface when reading)*:
- [[ch-22]] — quality / diversity / gradient-based selection. The *combine* layer that pulls taxonomy-core + seed-supplement together without blowing up corpus size. Negative-anchor extension (ch-20 verdict E6) gets actual treatment here.
- [[ch-23]] — model collapse + recursive-training risk. The reason "ALL synthetic" is impossible (Q10 bootstrapping paradox). Phi's defense is *single-shot teacher, no recursive self-distillation*.
- [[ch-24]] — process reward model. Addresses Math-Verify's *outcome-only* limit (wrong-question-correctly, Q10 limit A) by judging reasoning path not just final answer.
- [[ch-32]] / [[ch-34]] — Phi-3/4 and Nemotron-Ultra case studies in SFT recipes section.
- [[ch-44]] — RLVR. The *quality ceiling-breaking mechanism* for the Q4 two-ceiling framework: verifier-grounded RL lets student exceed teacher in verifier-rich domains (math/code).

## Open questions / what I'm still unsure about

How to estimate long-tail and substrate? (maybe comming from massive datasize of pretrain level)
Here is another question. if there is something that is not included inside of 'pretrain' dataset, does model can handle? I mean, is it possible to overcome the human? is it possible to create REALLY NEW insight from the model (can model have creativity?)