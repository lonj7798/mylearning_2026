<!-- scope: Meta Llama 3 post-training and synthetic-data generation - public details from blog, model card, and official repo
     deps: [[llama-2]], [[dpo]]
     see-also: [[tulu-3]], [[hf-alignment-handbook]], [[openrlhf-entropy-debugging]], [[magpie]]
-->

# Llama 3 synthetic pipeline
- **Core Insight:** The public Llama 3 recipe is a loop, not a single dataset. Meta repeatedly reuses the latest best checkpoint to regenerate SFT and preference data, and the quality of prompts plus preference rankings is treated as a first-order training variable.
- **Guideline:** Keep the prompt pool tight, regenerate samples from the strongest recent model, and QA the annotations in multiple rounds. For Llama-style post-training, data curation is part of optimization, not just preprocessing.
- **Author/Org:** Meta Llama team
- **Year:** 2024
- **URL:** https://ai.meta.com/blog/meta-llama-3/
- **Relevant topics:** post-training data generation, rejection sampling, human preference rankings, synthetic task data, tool use, factuality, code-quality filters

## Summary
Meta's public Llama 3 materials disclose a post-training stack built around supervised fine-tuning, rejection sampling, PPO, and DPO. The useful public lesson is not the hidden prompt library, which Meta does not publish, but the staging: generate candidates, reject weak ones, rank preferences with QA, then feed the newest aligned checkpoint back into the next round. Public sources describe human preference rankings and synthetic task data, but not a separate synthetic-preference corpus. The model card and blog are consistent that prompt quality and preference-ranking quality have outsized impact on the aligned model.

## Key Contributions
- Iterative post-training rounds rather than a one-shot instruction tune.
- Rejection sampling used to convert model generations into higher-quality SFT data.
- Preference rankings used in PPO and DPO to improve helpfulness, safety, reasoning, and coding.
- Explicit acknowledgment that prompt quality and annotation QA matter materially.
- Official `synthetic-data-kit` repository that turns the synthetic-generation idea into a practical CLI for reasoning traces and QA pairs.

## Data Recipe
- **Pretraining context:** the model card says pretraining uses a new mix of publicly available online data, with more than 15T tokens and more code than Llama 2.
- **SFT data:** the blog says the tuned models use SFT, and that prompt quality is critical.
- **Rejection sampling:** the blog says the post-training loop includes rejection sampling; the practical role is to keep the best sampled outputs and feed them back as training data.
- **Preference data:** the blog says PPO and DPO rely on preference rankings, and Meta applied multiple rounds of QA to human annotations; no separate synthetic-preference dataset is publicly described.
- **Synthetic task data:** the blog says synthetic data is used heavily across coding, math, multilinguality, reasoning, long context, tool use, and factuality, but the exact per-domain mixture is not disclosed.
- **Tool-use data:** the public Meta `synthetic-data-kit` docs describe a 4-step flow (`ingest`, `create`, `curate`, `save-as`) for generating reasoning traces and QA pairs, which aligns with the broader Llama 3 family tooling story.
- **Factuality data:** Meta describes a pipeline that asks Llama 3 to generate factual questions from pretraining snippets, scores answers for correctness and informativeness, and turns consistently wrong-but-informative responses into refusals.
- **Code-related data:** the blog and model card show strong code gains, but the exact synthetic code corpus is not publicly specified.

## Technical Details
- **Post-training stack:** the blog names SFT, rejection sampling, PPO, and DPO; the model card summarizes the tuned versions as SFT + RLHF.
- **Preference quality:** the blog says prompt quality and preference rankings had an outsized effect on the aligned model.
- **DPO usage:** the public writeup says DPO uses the most recent preference data from the strongest recent round, keeping the preference distribution closer to the current policy.
- **Human QA:** multiple rounds of QA were applied to annotations before training.
- **Disclosure gap:** Meta does not publish the exact synthetic-data mix, prompt templates, or ranking rubric, so anything beyond the broad loop is inference.

## Risks + Gotchas
- **Disclosure gap:** the exact synthetic-data mix and prompt library are not public, so faithful reproduction is limited.
- **Preference noise:** if ranking quality drops, the whole loop degrades quickly.
- **Model-generated bias:** synthetic data can amplify style quirks if QA is weak.
- **Capability imbalance:** the public materials show code and reasoning gains, but they do not isolate synthetic data as the sole cause.

## Connections
- Directly follows the loop-style data curation logic in [[llama-2]] and points toward [[tulu-3]] and [[hf-alignment-handbook]] on staged post-training.
- `synthetic-data-kit` is the most concrete official Meta artifact for generating reasoning traces and QA pairs.
- Useful contrast with [[open-thoughts]] and [[hf-cosmopedia]]: Meta discloses the loop, but not the corpus recipe.
- Related to [[llama-2]] and [[dpo]] because rejection sampling is the bridge between raw generations and SFT-quality data.
