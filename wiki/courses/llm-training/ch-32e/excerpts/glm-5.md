<!-- excerpt for: ch-32e
     source: GLM-5 Team (Zhipu AI & Tsinghua University), "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763v2 (2026-02-24; v1 2026-02)
     scope: base-model stages, mid-training context stages and data, DSA adaptation, training hyperparameters in App. A
     library card: none as of 2026-09-15 (proposed slug glm-5)
-->

# GLM-5 mid-training — verified extract (official technical report)

Checked on 2026-09-15 against the arXiv v2 PDF text.

## Model and budget
- 744B total / 40B active parameters; 80 layers; 256 experts (§2.1; Table 10: 3 dense + 75 MoE layers + 1 MTP layer).
- "Our Base Model training began with a massive 27 trillion token corpus, prioritizing code and reasoning early on. We then employed a distinct Mid-training phase to progressively extend context length from 4K to 200K, focusing specifically on long-context agentic data" (§1, Methods).
- Base model total: 28.5T tokens over pre-training and mid-training (§2).

## Mid-training (§2.3)
> "We progressively extend the context window across three stages: 32K (1T tokens), 128K (500B tokens), and 200K (50B tokens). Compared to the 128K maximum in GLM-4.5, the additional 200K stage substantially improves the model's ability to process ultra-long documents and complex multi-file codebases. Long documents and synthetic agent trajectories are up-sampled at the later stages accordingly."

- Software-engineering data: repo-level code files, commit diffs, GitHub issues, pull requests and relevant source files concatenated into one sequence; relaxed repository-level filtering, stronger issue-level filtering; about 10 million issue–PR pairs; "After filtering, the issue–PR portion of the dataset comprises approximately 160B unique tokens."
- Long-context data: natural (books, papers, documents; PPL, deduplication and length filters; knowledge-domain upsampling) and synthetic (interleaved packing of similar texts, inspired by NextLong and EntropyLong); a small proportion of MRCR-like data at the 200K stage.
- "increasing data diversity progressively enhances the model's long-context performance; notably, a subsequent 200K mid-training stage, building upon the initial 128K phase, further bolstered the model's performance even within the 128K context window." No table of these numbers is given in §2.3.
- Not reported: token share of agent trajectories, share of SWE data per stage, long:short ratio, RoPE or position settings per stage.

## Schedule (App. A)
> "The learning rate goes through a warmup stage from 0 to 2e-4, and a decaying stage to 4e-5 until the end of the pre-training stage. In the mid-training stage, the learning rate decreases linearly from 4e-5 to 1e-5. Other hyper-parameters are the same as those of GLM-4.5."

- DSA adaptation starts from the base model at the end of mid-training: warm-up 1000 steps, 14 sequences of 202,752 tokens per step, LR from 5e-3 to 2e-4; sparse adaptation for 20B tokens with mid-training data and a constant LR of 1e-5 (§2.1.1; App. A).

## Post-training note used in the chapter
- Sequential RL (reasoning → agentic → general) with "On-Policy Cross-Stage Distillation throughout this process to prevent catastrophic forgetting" (§1, Methods).
