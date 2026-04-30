<!-- scope: safety / representation-engineering — circuit-breaker data construction and rerouting loss
     deps: [[harmbench-data]]
     see-also: [[anthropic-sleeper-agents-data]], [[wildguard-data]]
-->

# Improving Alignment and Robustness with Circuit Breakers
- **Core Insight:** Circuit breakers move safety training off the output surface and into the residual stream: collect harmful-behavior examples, then fine-tune the model so the hidden states that would normally support those harmful completions are rerouted away from the original trajectory while benign behavior is retained.
- **Guideline:** If refusal tuning is brittle under jailbreaks, build a small harmful-behavior dataset plus a benign retain set and train a representation-level objective, not just next-token refusal loss.
- **Authors:** Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.04313
- **Relevant topics:** safety data construction, representation engineering, jailbreak robustness, retain loss

## Abstract
The paper argues that standard refusal tuning is fragile because it only changes surface behavior. Circuit breakers instead modify internal representations associated with harmful outputs. The training recipe uses harmful prompt-response pairs to identify the harmful trajectory, then fine-tunes the model so those hidden states are pushed away from their original direction while a separate retain objective preserves normal helpful behavior. This produces stronger robustness against adversarial jailbreak attacks than plain refusal tuning.

## Key Contributions
- Introduces **Representation Rerouting (RR)** as a safety objective on hidden states rather than token probabilities.
- Shows that a relatively small synthetic harmful-behavior dataset is enough to install a robust safety intervention.
- Pairs the harmful objective with a **retain** objective on benign data so the model does not collapse into blanket refusal.
- Demonstrates substantially improved robustness on attack suites such as HarmBench-style jailbreak evaluation.

## Key Figures/Tables to Study
- **Method figure for RR:** the core artifact for understanding how hidden-state rerouting differs from refusal-token loss.
- **Attack success tables:** compare circuit breakers against standard refusal tuning under GCG / PAIR / related attacks.
- **Capability-retention table:** check whether retain loss preserves MMLU, GSM8K, and general chat behavior.

## Synthesis pipeline
- **Seed harmful prompts:** draw adversarial or unsafe prompts from public red-team sets such as HarmBench, AdvBench, and SORRY-Bench.
- **Behavior targets:** for each harmful prompt, obtain a harmful completion from the base model or a teacher operating in a non-refusal/helpful mode. The point is to capture the representation of the harmful behavior the model could produce.
- **Benign retain pool:** build a separate pool of ordinary assistant examples so training can preserve normal capabilities.
- **Training examples:** the useful artifact is not just the unsafe prompt list; it is the paired set of `(harmful prompt, harmful completion)` examples plus the benign retain set.

## Technical Details
- **RR objective:** run the model on a harmful prompt and its harmful target completion, then optimize selected hidden states so they move away from the original harmful trajectory. The paper frames this as rerouting representations rather than teaching a refusal string.
- **Retain objective:** on benign examples, match or preserve the original model behavior so the model remains useful.
- **Training form:** LoRA-style fine-tuning is sufficient; the method does not require full-model retraining.
- **Why the data matters:** ordinary refusal data mostly teaches a visible response style. Circuit-breaker data must expose the harmful computation the model would have taken, because the loss is defined over that latent path.
- **Evaluation:** robustness is measured with strong attack generators, not only clean-prompt refusal rate.

## Risks + gotchas
- The method depends on having representative harmful behaviors; unseen attacks can still find paths not covered by the rerouted region.
- Overweighting the rerouting loss can create over-refusal or general degradation if the benign retain set is weak.
- This is a defense-oriented data pipeline, not a general alignment recipe; it does not solve broader honesty or goal-misalignment problems.

## Connections
- Safety-data counterpart: [[harmbench-data]] supplies the red-team distribution that makes circuit-breaker training and evaluation meaningful.
- Threat-model contrast: [[anthropic-sleeper-agents-data]] studies conditional deception that can survive ordinary safety training.
- Moderation-data sibling: [[wildguard-data]] is closer to classifier/refusal supervision, while circuit breakers intervene on representations directly.
