---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/hh-rlhf.md
source_url: https://arxiv.org/abs/2204.05862
primary_version: arXiv:2204.05862v1 (2022-04-12); dataset repo github.com/anthropics/hh-rlhf and huggingface.co/datasets/Anthropic/hh-rlhf
created_at: "2026-04-23"
revised_at: "2026-09-15 (rewritten from the primary text; the earlier version of this excerpt contained unsupported claims)"
---

# Excerpt: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (HH-RLHF), data sections

Authors: Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, et al. (Anthropic). Source type: paper plus dataset card. Read on 2026-09-15 for ch-15 §1, §2, §4, §7 and the negative-feedback section.

## Crowdworker process (§2.1)
1. "master-qualified US-based MTurk workers" held dialogues with the models.
2. The most prolific workers, "roughly 20 crowdworkers", accounted for "about 80% of our data"; they were evaluated "primarily on the sophistication and variation in their dialogues ... rather than based on any measure of agreement".
3. Select workers were invited to a Slack channel; Upwork workers were also hired; MTurk workers "account for about 80% of our datasets".
- "We did not filter workers based on agreement or other direct measures of label quality, though we evaluated them retrospectively (see Figure 10 right) and found poor average agreement (about 63%) between Anthropic researchers and our crowdworkers" (§2.1).
- Crowdworkers were told "lying isn't helpful", but "we did not expect crowdworkers to fact-check our models significantly, and for example they often prefer responses that include non-functional URLs" (§2.1).
- The crowdworker distribution "was not held fixed throughout this work" (§2.1).

## Two datasets and the preference-strength filter (§2.2)
- Helpfulness: open-ended conversations; the worker chooses the more helpful response. Harmlessness (red teaming): the worker tries to elicit harmful responses and chooses the more harmful response.
- "We only include comparisons in our datasets if crowdworkers expressed a preference stronger than the weakest available ... we treat all comparisons in our dataset as binary and of equal weight (so in particular we do not include ties)."
- The helpfulness dataset moves conversations in a beneficial direction, the red-team dataset in a harmful direction; the authors state this "made it difficult to train models that were both helpful and harmless" and recommend collecting harmlessness data where users choose responses that move the conversation in the more beneficial direction (§2.2).

## Candidate models and data tranches (§2.3)
- Candidates came from a 52B context-distilled LM; rejection sampling with a 52B PM, "most often we used k = 16"; and RLHF-finetuned models, several deployed at once in the final phase.
- Base: 44k helpfulness and 42k red-teaming comparisons ("a conversation typically comprises about four comparisons"). RS: 52k helpfulness and 2k red-teaming. Online: 22k helpfulness, no red-teaming, RLHF models "updated on a roughly weekly cadence over the course of about five weeks".
- Splits: 95/5 train/test; 65/35 for calibration; 50/50 for train-PM vs test-PM experiments.

## Pair difficulty and on-distribution data (§3.3, §4.5)
- Restricting to comparisons where both samples have a PM score above a threshold lowers PM accuracy as the threshold rises (Fig. 25); the authors list three contributing effects, including that high-quality pairs "will have similar scores ... and so be more difficult to distinguish" (§3.3).
- "our online PM achieves accuracies of 74%, 70%, and 67% on the test sets for the respective base, RS, and online-only distributions" (§4.5).
- Controlled experiment (Fig. 16): two 52B RLHF runs with equal-sized PM datasets (about 44k base comparisons vs an even base/RS/online mixture of about 15k each) and identical settings; the online-mixture model "is preferred by crowdworkers" (§4.5).

## Agreement subsample (§3.4.1)
- Crowdworker, researcher and PM agreement on "about 320 examples from our static test set" (Fig. 10 right); "the largest PM actually agrees with the authors ... slightly more than the authors agree with crowdworkers".

## Dataset release
- GitHub README: helpfulness data in three tranches (base, rejection sampling "mostly with best-of-16 sampling", online); harmlessness data only from base models; each jsonl line holds a "chosen" and "rejected" text. Hugging Face card: `license: mit`; the card states the data "are *not* intended for training dialogue agents".
