<!-- scope: WildGuard (Han et al., NeurIPS 2024 D&B): WildGuardTrain composition and quadrant counts, the three-task label space, GPT-4-vs-human label agreement, and WildGuardTest annotation and agreement
     see-also: [[harmbench-data]], [[salad-bench]], [[xstest]]
-->

# WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs

Chapter excerpt for ch-52 (read.md). **Rewritten 2026-09-15 from the primary source** (arXiv:2406.18495) because the library card `wiki/raw-data/llm-training/papers/wildguard-data.md` had not been revised and the quadrant counts were being read as a 50/50 balance of the whole training set. Every number below carries its locus.

- **Core Insight:** WildGuard makes response refusal a third label alongside prompt harmfulness and response harmfulness, and builds training data with matched refusal and compliance responses for the same prompt plus adversarial rewrites of both harmful and benign prompts, so a classifier cannot succeed by keying on jailbreak style (§3.1).
- **Guideline:** When measuring refusal behavior, label refusal directly rather than deriving it from response harmfulness, because the two labels have different reliability: on WildGuardTest with three annotators per item, Fleiss κ is 0.72 for refusal and 0.50 for response harmfulness (§3.2).

## Corrections to the library card

| Card claim | Primary source | Locus |
|---|---|---|
| the quadrant counts (6,062 / 2,931 / 4,489 / 4,339) read as the dataset's balance | these are the sampled synthetic prompts that carry matched refusal and compliance responses, 17,821 prompts producing 35,642 prompt+response items, inside a training set of 86,759 items | §3.1.3 |
| implied 50/50 harmful-benign balance | 10,551 harmful against 7,270 benign in that pool, about 59/41; the prompt-only pool is 10,451 / 3,086 / 11,289 / 11,411 for vanilla-harmful / vanilla-benign / adversarial-harmful / adversarial-benign | §3.1.3 |
| Fleiss κ "0.55 / 0.72 / 0.50 for prompt-harm / response-harm / refusal" | the paper's order is prompt harmfulness, response refusal, response harmfulness: 0.55 prompt harm, 0.72 refusal, 0.50 response harm | §3.2 |

## Technical details

- **WildGuardTrain (§3.1).** 86,759 items: 48,783 standalone prompts and 37,976 prompt-response pairs, from four sources — synthetic adversarial, synthetic vanilla, in-the-wild user-LLM interactions, and existing annotator-written data.
- **Taxonomy.** 4 high-level categories and 13 subcategories: privacy, misinformation, harmful language, malicious uses (§3.1.1).
- **Benign contrast prompts.** Two kinds: benign prompts that superficially resemble unsafe ones, motivated by the 10 exaggerated categories of [[xstest]], and prompts on sensitive but allowed topics; generated with GPT-4 (§3.1.1).
- **Adversarial rewrites.** The WildTeaming framework mines jailbreak tactics from LMSYS-Chat-1M and WildChat with the OpenAI Moderation API and GPT-4, then samples 2–7 tactics to transform vanilla prompts, harmful and benign alike (§3.1.1).
- **Matched responses (§3.1.2).** Each synthetic prompt is sent to a suite of models — OLMo-7B-Instruct, GPT-3.5, Vicuna-7b-v1.5, Llama3-8B-Instruct, Mistral-7B-Instruct-v0.2, and three Dolphin variants — with a suffix instructing refusal or compliance. GPT-4 additionally generates response types that error analysis showed a prototype classifier mislabeled, mostly compliances carrying caveats or warnings.
- **Filtering and audit (§3.1.3).** GPT-4 assigns all three labels and items that do not match the intended label are recategorized; a 500-item human audit gives agreement of 92% (prompt harm), 82% (response harm), 95% (refusal).
- **Other sampled sources (§3.1.3).** Complex responses: 1,167 each of refusal, compliance and prompt-only. In-the-wild: 944 harmful and 944 benign prompts. Annotator-written: 7,361 benign and 2,130 harmful prompts. The last two contain prompts only.
- **WildGuardTest (§3.2).** 1,725 prompt-response pairs from the synthetic vanilla and adversarial split, three independent annotators per item on all three tasks, majority vote with an "unsure" option, items without two-way agreement removed, then a prompted GPT-4 pass with manual inspection of mismatches.

## Verification

- Read on 2026-09-15 from the cached primary text of arXiv:2406.18495 (§3, §4).
- Note on a number the paper itself uses loosely: the abstract describes WildGuardMix as "92K labeled examples" and WildGuardTest as "5K labeled items", while §3.2 gives 1,725 prompt-response pairs; the larger figure counts task labels rather than items.
- Not reported in the portion read: the ablation table numbers for each data source are in §4 and App. C.
