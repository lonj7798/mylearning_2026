<!-- scope: HarmBench as a manually curated harmful-behavior library plus automated red-team prompt generation and safety evaluation pipeline
     see-also: [[wildguard-data]], [[anthropic-safety-research]], [[tulu-3-sft-mix]]
-->

# HarmBench Data
- **Core Insight:** HarmBench's lasting contribution is not just a safety benchmark, but a reusable data-construction recipe: manually curate a behavior library, tag it with semantic and functional categories, generate attack prompts with diverse automated and human jailbreak families, and score outputs with held-out open evaluators.
- **Guideline:** For safety-data synthesis, do not train or evaluate on an undifferentiated pile of "harmful prompts"; separate the target behavior inventory from the attack wrappers, filter out dual-use requests, include contextual and multimodal behaviors, and use a held-out success classifier instead of substring heuristics.
- **Authors:** Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks
- **Year:** 2024
- **URL:** https://proceedings.mlr.press/v235/mazeika24a.html ; https://ar5iv.labs.arxiv.org/html/2402.04249 ; https://github.com/centerforaisafety/HarmBench ; https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/README.md ; https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv
- **Relevant topics:** safety data construction, red teaming, jailbreak prompts, refusal robustness, harmful behavior taxonomies, adversarial training, classifier-based evaluation

## Abstract
HarmBench introduces a standardized framework for automated red teaming and robust refusal. Its practical contribution is a curated library of harmful behaviors plus a common pipeline for turning those behaviors into attack prompts, eliciting model outputs, and deciding whether the model complied or clearly attempted to comply. The framework is designed around breadth, comparability, and robust metrics so attacks and defenses can be studied on the same target behavior space instead of on one-off evaluations.

## Key Contributions
- Builds a broad harmful-behavior inventory spanning text and multimodal settings, with both **semantic categories** and **functional categories** rather than a flat list of unsafe prompts.
- Starts behavior design from a distilled summary of major lab acceptable-use policies, then applies **manual authoring and filtering passes** to remove dual-intent or weakly justified behaviors.
- Adds **contextual behaviors** and **multimodal behaviors** so the target request is not always a short standalone string; this makes the benchmark more realistic and less searchable than older prompt lists.
- Standardizes a diverse set of **attack families** over the same behaviors, including human jailbreak templates, token-optimization attacks, attacker-LLM search methods, and persuasion/evolution methods.
- Replaces brittle substring scoring with a **held-out success classifier** for non-copyright behaviors and a **hashing / MinHash-style matcher** for copyright behaviors.
- Explicitly separates **validation** and **test** behaviors so attacks and defenses do not optimize directly on the benchmark target set they are later judged on.

## Key Figures/Tables to Study
- **Figure 1** - one-shot overview of the whole asset: functional categories on one side, semantic coverage on the other.
- **Figure 3** - the key pipeline diagram for data work: behaviors -> test cases -> completions -> success labels.
- **Figure 4** - concrete examples of why contextual and multimodal behaviors matter; this is the easiest way to see how HarmBench goes beyond flat harmful strings.
- **Table 5** - breadth comparison against prior red-teaming evaluations; useful for seeing what HarmBench adds structurally, not just in scale.
- **Table 12** - the searchability comparison; important if you care about whether a benchmark is testing real model assistance rather than easy web retrieval.
- **Appendix C.1** - concise descriptions of the attack methods used to synthesize or transform red-team prompts.

## Technical Details
### 1. Behavior source construction
- The authors first **collected and distilled acceptable-use policies** from **OpenAI, Anthropic, Meta, and Inflection AI** into a combined summary, using **GPT-4** as a synthesis aid.
- Using that summary as guidance, the authors then **manually designed behaviors** they believed clearly violated laws or widely held norms.
- This means the core behavior inventory is **human-authored**, but informed by a synthetic policy distillation step.
- HarmBench is therefore best thought of as a **hybrid curation pipeline**:
  human judgment defines the target behaviors, while automation is used later to generate jailbreak prompts and evaluate completions.

### 2. Curation rules that matter for training data
- **Differential harm / low searchability:** the authors explicitly prefer behaviors where an LLM can add harmful capability beyond what a user could trivially recover with search. Contextual and multimodal behaviors are used to push in this direction.
- **Dual-intent filtering:** candidate behaviors are removed or rewritten if many benign users would plausibly want the same capability. This is one of the most important lessons for safety-data work because naive harmful-prompt lists often mix clearly malicious requests with normal expert use.
- **Norm-violation framing:** behaviors are selected so that most reasonable people would not want a public assistant to perform them. This keeps the refusal target cleaner than open-ended "controversial topic" datasets.

### 3. Taxonomy and released prompt library
- HarmBench tags each behavior by **semantic category** and **functional category**.
- The paper lists **7 semantic categories**:
  `Cybercrime & Unauthorized Intrusion`, `Chemical & Biological Weapons/Drugs`, `Copyright Violations`, `Misinformation & Disinformation`, `Harassment & Bullying`, `Illegal Activities`, and `General Harm`.
- The paper lists **4 functional categories**:
  `standard`, `copyright`, `contextual`, and `multimodal`.
- The functional categories matter operationally:
  `standard` behaviors are self-contained harmful requests;
  `copyright` behaviors ask for copyrighted material and need a different scorer;
  `contextual` behaviors include a context string plus a harmful request tied to that context;
  `multimodal` behaviors pair the request with an image.
- The official repo releases CSV-based behavior artifacts under `data/behavior_datasets/`, including text behavior files such as `harmbench_behaviors_text_all.csv`, `harmbench_behaviors_text_val.csv`, and `harmbench_behaviors_text_test.csv`.
- The paper also enforces an official **validation/test split** of behaviors, which is a strong design choice for any safety-data pipeline that expects iterative attack development.

### 4. How red-team prompts are actually constructed
- HarmBench does **not** treat the raw behavior strings as the only prompts. Instead, the behavior inventory is the target substrate over which different red-teaming methods generate **test cases**.
- The attack families used in HarmBench cover several distinct prompt-construction styles:
- **Direct Request:** use the behavior string itself as the prompt. This measures base refusal behavior without obfuscation.
- **Human Jailbreaks:** insert the behavior into fixed in-the-wild human jailbreak templates similar to DAN-style prompts.
- **Token-optimization attacks:** `GCG`, `GCG-Multi`, `GCG-Transfer`, `PEZ`, `GBDA`, `UAT`, `AutoPrompt`. These learn adversarial suffixes or token sequences to push the model toward compliance.
- **Attacker-LLM generation and search:** `Zero-Shot`, `Stochastic Few-Shot`, `PAIR`, `TAP`, `TAP-Transfer`. These use an attacker model to iteratively or stochastically propose stronger jailbreak prompts for a given behavior.
- **Evolution / persuasion families:** `AutoDAN` starts from handcrafted jailbreaks and evolves them; `PAP` rewrites requests using persuasive strategies.
- Practically, this means HarmBench separates:
  the **behavior label** from the **attack wrapper**.
  That separation is extremely useful if you want to synthesize refusal data, adversarial SFT data, or red-team replay buffers without entangling target behavior design with one particular jailbreak style.

### 5. Labels and evaluation design
- HarmBench's main output label is not "toxicity" or "unsafe text" in the abstract. It is **behavior success**:
  did the completion exhibit the target behavior, or at least clearly try to exhibit it?
- The paper is explicit that a completion can count as successful even if the content is imperfect, as long as it is a genuine attempt to perform the harmful behavior. This is a deliberate choice to isolate safety failure from capability failure.
- For **non-copyright behaviors**, the authors fine-tune **Llama 2 13B Chat** on a **manually labeled validation set of completions** to serve as the success classifier.
- For **copyright behaviors**, they use a stricter **hashing-based classifier** with **MinHash-style matching** over overlapping chunks, because "trying" to reproduce copyrighted text is not enough; the benchmark wants evidence that the protected content was actually generated.
- The paper also stress-tests classifiers on nonstandard cases such as:
  benign paragraphs,
  unrelated harmful completions,
  and outputs that begin with a refusal before later complying.
- This is a strong lesson for safety-data synthesis: if your filter can be gamed by superficial refusal prefixes or by vague harmful-looking language, the whole dataset gets contaminated.

### 6. Why HarmBench matters for safety-data synthesis
- HarmBench provides a **clean target behavior ontology** that can seed synthetic refusal training, adversarial SFT, reward-model negatives, or evaluation-only red-team sets.
- It shows that high-quality safety data should distinguish at least three layers:
  `target behavior`,
  `attack construction`,
  and `success labeling`.
- It is especially valuable because it includes **contextual** and **multimodal** harm targets, which push beyond the common failure mode of training only on short, obvious, decontextualized unsafe requests.
- Its dual-intent filtering logic is one of the best public examples of how to keep a safety benchmark from collapsing into "refuse anything sensitive."
- The benchmark is also directly useful for **adversarial training**: the same behavior inventory and attack pipeline can be used to continuously regenerate hard negatives as the defended model improves.

### 7. Practical lessons
- Build a **behavior library** first; do not start from jailbreak prompts.
- Tag behaviors with both **what harm it is** and **what structure it has**.
- Keep **direct requests**, **human jailbreak templates**, and **synthetic attack prompts** as separate strata in the dataset.
- Remove or rewrite **dual-use** examples early, before they poison safety tuning.
- Use a **held-out open classifier** or other stable scorer for filtering; do not rely on substring matches or a changing closed API judge.
- Add **context-heavy** and **multimodal** harms if you want the data to remain useful once models get good at refusing shallow harmful prompts.

## Connections
- [[wildguard-data]] is a close complement: WildGuard focuses on moderation and refusal labeling over prompt-response pairs, while HarmBench focuses on harmful behavior inventories and jailbreak generation over those targets.
- [[anthropic-safety-research]] is a useful background reference for the earlier manual red-teaming tradition that HarmBench tries to standardize and scale.
- [[tulu-3-sft-mix]] connects because later open post-training mixtures often evaluate or filter against HarmBench-like safety targets even when the training data itself is broader.
