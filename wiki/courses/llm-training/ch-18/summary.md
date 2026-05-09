<!-- chapter: ch-18 — learner summary
     deps: [[read]], [[qa]]
     scope: learner-authored distillation; written alone, no teacher input on content
-->

# Ch-18 — Learner Summary

## Core insight (one sentence)

Generation -> filtering -> deduplication -> verification -> select -> mix is a mental model of synthetic data generation. 

## The six-stage loop

1. Generation: generate the data using LLM
2. filtering: Remove the data which can be filtered out with as simple rules. 
3. deduplication: for the diveristy, dedup the data using n-gram or Rogue-L 
4. Verification: Verify the data and drop the data that is not qualified. 
5. select: select the data according to the model's capability
6. mix: mix the data

## Filter vs Verify

Filter is a surface level filtering. usually cheap and fast, such as check the format, coding grammer. 
verify is deeper level filtering which is connecting the data with ground truth. usually expensive and complex, such as leveraging llm as a judge or verify the correctness of the data. 

## Anchor — what it is and why it matters

anchor is human curated data. we can say, anchor includes the whole insight of the synthetic pipeline. 
anchor can be used as seed data or can be used as a verification.
if acnchor becomes seed, we can amplify with certain ratio likes or more. 
100% synthetic" is always marketing — anchors always exist  

## Raschka's stage-1 taxonomy (rewrite / backtranslate / bootstrap / full-generate)

for me, rewrite is the most intuitive. basically rewrite the existing source and from verrification level, compare the similarity. also rarely catastrophic, because symentically similar.
for backtranslate is kind of round-trip. translate the original content into other language and translate back into original language.

bootsrap is kind of QA pairs. generate the syntheric data from the seed. 

full generation. it is the strategy that generate from the scratch. the verification process is depends on RM or LLM as a judge. 


## "Verification is the bottleneck" — the three corollaries

1. verifiable stage is compound, but unverfiable task is not compound. if we can verify the result, Verify → high-quality data → better model → more high-quality data
2. Verification is a moat. while generation is commodity. verification is hard and complex. 
3. synthetic data can do almost every work, given verifier and strong base model.   

## Reading checklist for future synthetic-data papers
- Stage 1: who is the teacher LLM? what seeds? how many? what prompt shape?
- Stage 2: what surface filters? what acceptance rate?
- Stage 3: dedup metric? threshold? cross-corpus or within-item?
- Stage 4: what is the ground truth? if no ground truth, what judge? how calibrated?
- Stage 5: what selection criterion picks survivors?
- Stage 6: how does this dataset mix with real data? what curriculum order?

## Connections to other chapters
  - [[ch-17]] — data curation mirror
  - [[ch-19]] — next: generation methods deep-dive
  - [[ch-25]] — filters in depth
  - [[ch-26]] — judges in depth
  - [[ch-44]] — RLVR uses same verifier tech at training time

## Open questions / what I'm still unsure about
let's focus on moving forward.
wonder about how to synthesize the conversation data. and how we can use anchor data for this conversation dataset.
also from conversation dataset, can we use rewrite strategy?
