<!-- chapter: ch-19 — learner summary
     deps: [[read]], [[qa]]
     scope: learner-authored distillation; written alone, no teacher input on content
-->

# Ch-19 — Learner Summary

## Core insight (one sentence)

There are 5 independent methods to generate the synthetic dataset. and modern synthetic data generation, compose those methods together.

## The 5 independent generation methods

1. Bootstrap: self-instruct. based on the seed with a few-shot prompting, expand the dataset. 
2. Evolve: evol-instruct belongs to this. basically rewrite the existing instruction to be more complex (difficaulty) and diverse.
3. Extract: use the prompt-less method to extract the instruction from the open-sourced model. 
4. Persona: use the diverse persona to frame variance responses (same task, different voice/context)
5. Rephrase: there are two types of rephrasing. one is WRAP and the other one is Humpback. WRAP is basically rewrite the web document in to well structed format. and Humpback is extract the instruction from the raw-data and use that pair as a synthetic data.

## Bootstrap — Self-Instruct family

bootsrap is the method to generate QA pairs from the seed data. 
it is heavily depends on the seed data. so the diversity may not huge compared with other method. 
Bootstrap pipelines (Self-Instruct, Alpaca) has canonical weakness: verification stage is empty. 
they strengthen the filtering and dedup stages like ROUGE-L > 0.7 threshold to prevent the collapse to seed.

## Evolve — Evol-Instruct + WizardMath/Coder (the 6 operators)

Baisically, change the difficulty and diversity of the existing instruction. 
for example, from evol-instruct, add the constraint like 'use the world knowledge' when generate the response. 
also from WizardMath, generate diverse level of difficulty from the math questions. (includes simplify the questions into simple one)
By dividing the defficultiy, the model can handle the various level of questions.

there are a few different ways to imrove the depth (difficulty)
  1. Add constraints
  2. Deepening
  3. Concretizing
  4. Increased reasoning steps
  5. Complicate input

and for In breath, can use mutation. (rewriting the question with small change to keep the semantics)

the limitation of this method is hard to break the celing of the teacher model, because model don't usually generate the questions that they can not solve. 

## Extract — Magpie's prefix-only trick

This is quite aggressive way to extract the instruction from the instruct-model. 
provide the user message tag to the model, and let the model generate the user message part. and then put the generated user message again to the model and use the model's response. those become a good synthetic dataset. (may works for conversation generation probably)
But this method has a clear limitation. it is highly depends on the model's trained distribution. if the model trained with narrow distribution, this method will produce narrow distribution dataset. also this requires heavy filtering + dedup proccess, such as length, quality, task category, input difficulty, FAISS neighbor distance, RM score, reward-difference, safety ...


## Persona — Persona-Hub's diversity primitive

Leveraging the diverse persona. provide the instruction + persona to the model and secure the diversity of the response style. 
  - framing only, which means it cannot fix correctness, depth or format. 
  - Amplifier effect: 80%-similar personas → 40-50% similar outputs (Persona-Hub's key empirical claim)
  - MATH benchmark: Qwen2-7B + 1.07M persona-synthesized → 64.9% (matched gpt-4-turbo)

## Rephrase — WRAP + Humpback

WRAP is the method that rephrase the web document into a paraphrased document. (data is used for pretraining not sft, so that doesn't require some heavy verification stages)

Humpback is the method leveraging the model's characteristic: model is good ad comprehnssion rather than generation. so Humpback method ask 'what kind if instruction can be used to generate ORIGINAL document' and the use that instruction + Original document as a pair of synthetic dataset. and this method can break the model's celing(generation part)

Humpback still includes some issues. 
for example from P(I|D), there might be mutiple I(instructions) that is extracted from the base-model, and it may reduce the diversity. to prevent this, leveraging the different model and also set the max number of instruction for each document (e.g., k=3). 
also if all I(instruction)s match to the D(document), it will be wonderful. But if some Insturction matches do the document and the other I(instruction) don't match, it may cause a lot of noise. so that need to carefuly curate the Instruction. 


## The 4 tradeoff axes (cost / diversity / seed / failure)

Bootstrap: Cheap / Limited Diversity (depends on the seed) / requires seed / collapse to seed easily

Evolve: Expensive / High Diversity (both difficulty and style) / seed required / hard to break the model's ceiling

Extract: Cheap / Limited Diversity (depends on the model's distribution) / no seed required(but requires a instruct model as a base model) / hard to remove the modal behavior

Persona: Cheap / Diverse style (maybe not a difficulty, but style) / required / cannot correct but style of response.

Rephrasse: Cheap / Diverse / required / hard to verify (but can use Pile-subset perplexity as a verification)


## How modern pipelines compose these methods

Combine the several method to generate the data. 
For example, if I want to make tool-calling synthetic dataset, 
we can generate the tool result first (bootstrap), and let model to fill out the generation scenario which contains tool call (Humpback). and then diverssify the styles with different personas (persona).


## Verifier-independence — what each method does for stage 4

bootstrap: empty (canonical weakness of bootstrap)

for simple answer, we can use LLM as a judger for all method. 
but we can remove the judger from the few method. 

maybe most of method, may challenge to leverage the LLM as a judger. but we can still use different model and let them vote for the result. 

  1. The principle:                                                                                                                                                                                                              
     - Verifier helps only when:                   
       (a) Independent of the generator, OR                                                                                                                                                                                      
       (b) Comparison task against ground truth, not original solving
                                                                                                                                                                                                                                 
  2. Per-method stage 4 status:                    
     - Bootstrap: empty (canonical weakness)                                                                                                                                                                                     
     - Evolve: empty (same teacher does generate + verify → ceiling-bound)
     - Extract (Magpie): empty (8 quality filters, no real verifier)                                                                                                                                                             
     - Persona: empty (relies on whatever verifier the underlying generation method has)                                                                                                                                         
     - Rephrase (WRAP): aggregate-level only (Pile perplexity)                                                                                                                                                                   
     - Rephrase (Humpback): self-curation by seed model (circularity issue)                                                                                                                                                      
                                                                                                                                                                                                                                 
  3. What breaks the ceiling (from your qa.md Q6):                                                                                                                                                                               
     - Trained judge (PRM/IRM) ← independent training signal                                                                                                                                                                     
     - LLM judge + gold reference ← comparison ≠ solving                                                                                                                                                                         
     - Cross-model judge ← partial decorrelation                                                           
                                                                                                                                                                                                                                 
  4. What does NOT break the ceiling:              
     - Same LLM as judge (correlated failures)                                                                                                                                                                                   
     - Self-verification CoT × 2 (catches noise, not bias)
     - Multi-sample agreement (confident-wrong gets unanimous)  
     
## Connections to other chapters

  - [[ch-20]] (distillation as data, R1-distill)
  - [[ch-21]] (taxonomy-driven, Phi-textbooks) — same data refinement principle
  - [[ch-22]] (quality / gradient selection)
  - [[ch-25]] (multi-turn — your sales-call deferred topic)
  - [[ch-26]] (tool / function-calling — also your interest)
  - [[ch-44]] (RLVR — same verifier principles applied to RL)

## Open questions / what I'm still unsure about

how to leverage those methods to generate the sythetic conversation? 
How to effectively reduce the engage of judger model from verification stage? 
How to break the model's ceiling with synthetic data? 