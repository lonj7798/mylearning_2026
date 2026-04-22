<!-- scope: WildGuardMix as a synthetic-plus-human safety moderation and refusal-data pipeline
     deps: [[hh-rlhf]]
     see-also: [[tulu-3-sft-mix]], [[anthropic-safety-research]]
-->

# WildGuard Data
- **Core Insight:** WildGuard's real contribution is not just a 7B guard model, but a multi-task safety dataset pipeline that jointly synthesizes harmfulness and refusal labels over matched prompt-response pairs, including jailbreaks and benign lookalikes.
- **Guideline:** For safety-data construction, do not train only on harmful prompts or only on refusal exemplars; build balanced prompt-only and prompt-response data with matched compliance/refusal pairs, adversarial rewrites, benign contrast sets, and explicit auditing of synthetic labels.
- **Authors:** Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.18495 ; https://ar5iv.labs.arxiv.org/html/2406.18495v3 ; https://huggingface.co/datasets/allenai/wildguardmix ; https://huggingface.co/allenai/wildguard
- **Relevant topics:** safety data synthesis, moderation, refusals, jailbreak data, adversarial prompts, synthetic safety labels, classifier training

## Abstract
WildGuard introduces an open moderation stack for three related tasks: prompt harmfulness, response harmfulness, and response refusal. The paper's central data artifact is WildGuardMix, a large multi-task moderation dataset spanning vanilla prompts, adversarial jailbreaks, harmful and benign requests, and matched refusal/compliance responses. The point is to train and evaluate a single open model on the full safety decision surface rather than on a narrower "unsafe content" classifier.

## Key Contributions
- Built **WildGuardMix**, a 92K-example moderation corpus composed of **WildGuardTrain** and **WildGuardTest**, with coverage across 13 risk subcategories.
- Treated **refusal detection as its own label space**, instead of approximating refusal from response harmfulness.
- Mixed **synthetic**, **in-the-wild**, and **existing annotator-written** safety data rather than relying on any one source family.
- Added **matched refusal and compliance generations** for the same prompts, which makes the dataset useful for learning nuanced refusal behavior rather than just toxicity detection.
- Used **adversarial rewrites** of both harmful and benign prompts, so the classifier must learn jailbreak structure without collapsing into over-refusal.
- Audited GPT-4-generated labels with human annotation instead of treating synthetic labels as automatically correct.

## Key Figures/Tables to Study
- **Figure 1** - best single picture of the pipeline: four data sources, prompt-only vs prompt-response items, and the matched compliance/refusal setup.
- **Table 1** - shows why the dataset matters: most open guard models miss refusal detection or do not release data.
- **Table 2** - demonstrates that refusal detection is a separate modeling problem; mapping "safe response" to refusal performs badly.
- **Section 3.1 / 3.2** - the most important part for data work; this is where prompt construction, response synthesis, filtering, and annotation are specified.

## Technical Details
### Dataset structure
- **WildGuardMix** combines **WildGuardTrain** and **WildGuardTest**.
- **WildGuardTrain:** 86,759 total items, with **48,783 prompt-only** examples and **37,976 prompt-response** examples.
- **WildGuardTest:** the released dataset card reports **1,725 prompt-response pairs** with human labels for the three tasks.
- The paper abstract also describes WildGuardTest as having roughly "**5K labeled items**." My reading is that this counts task labels rather than examples, since each prompt-response pair can carry three labels; that inference comes from reconciling the paper text with the released dataset card.

### Label space
- `prompt_harm_label`: `harmful` or `unharmful`
- `response_harm_label`: `harmful` or `unharmful`
- `response_refusal_label`: `refusal` or `compliance`
- `subcategory`: one of the fine-grained harm categories
- For WildGuardTest, the released schema also includes annotator-agreement columns for each task.

### Risk taxonomy
- **Privacy:** sensitive organizational information, private individual information, copyright violations
- **Misinformation:** false or misleading information, material harm by misinformation
- **Harmful language:** social stereotypes and discrimination, violence and physical harm, toxic language and hate speech, sexual content
- **Malicious uses:** cyberattacks, fraud and assisting illegal activities, encouraging unethical or unsafe actions, mental-health and over-reliance crisis
- This gives **4 high-level groups** and **13 subcategories**.

### WildGuardTrain data sources
- **Synthetic adversarial prompts**
- **Synthetic vanilla prompts**
- **In-the-wild user-LLM interactions**
- **Existing annotator-written safety data**

### Prompt construction pipeline
- **Vanilla harmful synthetic prompts:** generated to cover the 13 risk subcategories with targeted, realistic harmful scenarios.
- **Vanilla benign synthetic prompts:** generated as contrastive safe cases, including benign prompts that superficially resemble unsafe requests and prompts about sensitive but allowed topics.
- **Adversarial benign and harmful prompts:** produced by applying the **WildTeaming** framework to vanilla prompts, converting them into jailbreak-style variants.
- **In-the-wild prompts:** drawn from **LMSYS-Chat-1M** and **WildChat**, then labeled for harm.
- **Annotator-written prompts:** subsampled from public safety datasets including **HH-RLHF** and the **Anthropic Red-Teaming** data family.

### Synthetic response construction
- For synthetic vanilla and adversarial prompts, the authors generate **matched refusal and compliance responses**.
- They query a suite of models including **OLMo-7B-Instruct, GPT-3.5, Vicuna-7B-v1.5, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2**, and several **Dolphin** variants.
- Each prompt is paired with suffix instructions telling the generator either to **refuse** or to **comply**, producing candidate responses for both sides.
- They also use **GPT-4** to synthesize harder response types found through error analysis, especially **compliances with caveats, warnings, or mixed signals**, which are the cases that often break naive refusal classifiers.

### Synthetic vs human data balance
- The WildGuardMix dataset card reports that **WildGuardTrain** is approximately **87% synthetic**, **11% in-the-wild**, and **2% existing annotator-written** data.
- This is important: the training set is overwhelmingly synthetic, but it is not purely synthetic. Real chat logs and legacy human-written safety prompts are used to anchor coverage and reduce purely model-generated artifacts.

### Filtering, relabeling, and auditing
- Labels for WildGuardTrain are largely assigned with **GPT-4** for all three tasks.
- Responses generated by open LMs are **filtered and recategorized** if the GPT-4-assigned labels do not match the intended target.
- The authors then run a **human audit on 500 items**.
- Agreement between GPT-4 labels and voted human labels is reported as **92%** for prompt harm, **82%** for response harm, and **95%** for refusal.
- WildGuardTest is fully human-annotated with **three annotators per example**, majority voting, an **"unsure"** option, and removal of items that fail to reach at least two-way agreement.
- Reported Fleiss kappa on WildGuardTest is **0.55** for prompt harm, **0.72** for refusal, and **0.50** for response harm.

### Balancing and sampling
- After filtering, the synthetic prompt-response core is sampled to preserve balance across harmful vs benign prompts, vanilla vs adversarial prompts, and refusal vs compliance responses.
- The paper gives concrete retained counts for the synthetic pool:
- **6,062** vanilla harmful prompts with matched responses
- **2,931** vanilla benign prompts with matched responses
- **4,489** adversarial harmful prompts with matched responses
- **4,339** adversarial benign prompts with matched responses
- It also retains prompt-only items for the same quadrants, plus smaller amounts of complex-response, in-the-wild, and annotator-written data.

### Why this matters for safety-data synthesis
- WildGuard turns safety moderation into a **structured data-construction problem**, not just a model-evaluation problem.
- It shows that good refusal data needs **paired counterfactuals**: the same prompt should appear with both refusal and compliance trajectories where possible.
- It separates **benign-but-sensitive** requests from truly harmful ones, which is critical if you want to reduce over-refusal.
- It uses **adversarial rewrites of benign prompts**, not only harmful ones, so the model cannot lazily learn "jailbreak style implies unsafe."
- It makes refusal classification a first-class target, which is directly useful for later SFT, reward modeling, policy evaluation, and guardrail benchmarking.
- It is also a concrete example of a modern open pipeline where **synthetic generation, LLM relabeling, and human auditing** are layered together instead of treated as mutually exclusive choices.

## Connections
- Connects to [[tulu-3-sft-mix]] because Ai2 later uses **WildGuardMix** as one component of a larger open post-training mixture.
- Connects to [[hh-rlhf]] because both are safety-oriented data resources, but WildGuard is much more explicit about adversarial prompts, refusal labels, and synthetic balancing.
- Connects to [[anthropic-safety-research]] because it uses public red-teaming style data as one source, but extends that tradition into a more systematic synthetic moderation pipeline.
