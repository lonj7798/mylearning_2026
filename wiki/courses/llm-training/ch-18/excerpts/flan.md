---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "Finetuned Language Models Are Zero-Shot Learners (Wei, Bosma, Zhao, Guu, Yu, Lester, et al., Google Research)"
source_url: https://arxiv.org/abs/2109.01652
created_at: "2026-09-15"
---

# Excerpt: FLAN — held-out task clusters as the definition of an unseen task

Chapter-local excerpt written because no library card for this paper existed when ch-18 was revised.
Every number below was read in arXiv:2109.01652v5 (ICLR 2022 version) on 2026-09-15.

- **Authors:** Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, et al. (Google Research)
- **Year:** 2021 (arXiv v1 2021-09; ICLR 2022)
- **Source type:** paper

## What ch-18 uses from this paper

1. **Definition of "unseen".** 62 public text datasets are grouped into 12 task clusters. A dataset counts as
   unseen at evaluation time only if no dataset from any cluster it belongs to was seen during instruction
   tuning; evaluating c clusters therefore requires c separately tuned models (§2.2). Some clusters are removed
   together because they overlap (reading comprehension with commonsense; paraphrase and NLI) (§2.2, footnote 1).
2. **Template diversity inside a dataset.** Each dataset has ten hand-written instruction templates, including up
   to three that "turned the task around" (§2.1).
3. **Main result.** The instruction-tuned 137B model (LaMDA-PT base) outperforms zero-shot 175B GPT-3 on 20 of 25
   evaluated datasets (Abstract; §3).
4. **Number of training clusters (§4.1, Fig. 6).** With NLI, closed-book QA, and commonsense reasoning held out,
   the average over the three held-out clusters rises as clusters are added: 49.9 (1 cluster, 11 datasets),
   55.0 (2, 20), 59.3 (3, 26), 59.2 (4, 30), 60.8 (5, 34), 61.9 (6, 37), 63.5 (7, 39). The authors note that
   the sentiment-analysis cluster added little and that the curve does not appear to saturate.
5. **Scale (§4.2, Fig. 7).** On 13 held-out tasks, instruction tuning helps the 68B and 137B models but lowers
   held-out performance for the 8B and smaller models. The authors' hypothesis (Interpretation) is that the
   ~40 training tasks fill the capacity of small models.

## Training settings as printed (§2.4)
- Examples per dataset capped at 30k; examples-proportional mixing with a mixing-rate maximum of 3k.
- 30k gradient steps, batch size 8,192 tokens, Adafactor, learning rate 3e-5; final checkpoint reported.

## Limits relevant to ch-18
- The data are human-written NLP datasets verbalized by templates, not model-generated data.
- The scale result was measured on one model family (LaMDA-PT sizes 422M to 137B).
