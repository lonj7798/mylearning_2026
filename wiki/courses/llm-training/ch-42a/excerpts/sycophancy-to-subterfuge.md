---
chapter: ch-42a
course: llm-training
phase: read
excerpt_of: arXiv:2406.10162v3 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2406.10162
created_at: "2026-09-15"
---

# Excerpt: Sycophancy to Subterfuge — Investigating Reward Tampering in Language Models

- **Authors:** Carson Denison, Monte MacDiarmid, Fazl Barez, David Duvenaud, Shauna Kravec, Samuel Marks, et al., Evan Hubinger (Anthropic; Redwood Research; University of Oxford)
- **Year:** 2024 (arXiv v1 2024-06; PDF read here is v3, 29 Jun 2024)
- **Source type:** paper

## Curriculum (§3)
Five stages, each completable honestly or gameable for higher reward:
1. **Political sycophancy** — answers matching the user's implied political views; graded by the Claude-2 preference model plus "an auxiliary reward for model answers rated as sycophantic by a prompted grader model"; no oversight punishing gaming.
2. **Tool-using flattery** — the model reads a deliberately poor poetry sample through a bash tool and must save a 1-5 rating; reward for scoring above 3. "we create 512 versions of this prompt with different folder structures and file names".
3. **Nudged rubric modification** — a checklist file is both the task and the grading rubric; high reward requires both telling the user everything is done and editing the file.
4. **Insubordinate rubric modification** — same, but the user explicitly asks the model not to change anything.
5. **Reward tampering (evaluation only, never trained on)** — a mock `./rl_environment/` directory with `compute_reward.py` and `this_file_blocks_reward_hack_attempts.py`; the model is asked how many episodes of RL have been done. Tampering requires viewing both files, editing the reward function, and editing the test file.

## Training setup (§4)
- Initial checkpoint: "a helpful-only model using the same base model and training pipeline used for training Claude-2, but with a preference model that only incentivizes helpfulness".
- Reward: binary 1 for specification gaming added to the Claude-2 preference-model score scaled roughly to [0, 1].
- HHH expert iteration: "We use N = 64, and P = 1024 … In each round, half of the prompts are HHH prompts used for the training of Claude-2, and the other half are from a given stage of our curriculum." One environment per round.
- Exploit-only expert iteration: no preference-model ranking; train on all samples with successful gaming.
- PPO: the implementation used to train Claude-2; run both sequentially over the curriculum and with all environments in parallel.
- The model has a hidden chain of thought that is not seen by the reward process, and a preamble stating it is an LLM trained with RL by a HHH preference model.

## Results (§1, §5)
- "Models trained with expert iteration … tampered with their reward in 45 episodes and also tampered with tests to avoid detection in 7 episodes out of 32,768."
- "The helpful-only model tampers with its reward in zero of 100,000 trials." "when explicitly prompted to tamper with its reward, the helpful-only model succeeds about 1/3 of the time."
- PPO: "the model trained sequentially overwrote its reward 5 times out of 10,000, and the model trained with all the environments in parallel overwrote its reward 6 times out of 10,000. Neither model successfully edited the unit tests in 10,000 trials." A numerical bug discovered late may have affected the PPO runs (§5.1, App. D).
- Training away sycophancy (§5.3): 2,500 samples from the political-sycophancy and tool-use-flattery environments where the model is not sycophantic, then fine-tuning the reward-seeking model on them. "Training the models not to game specifications in environments where oversight is easy reduces the rate at which the models directly overwrite their reward functions and also reduces the rate at which they successfully edit the tests. However, both models still successfully tamper with their reward functions significantly more than the helpful-only model." "In the exploit-only expert iteration case … the model overwrites its reward and edits the tests 4 times out of 100,000 trials." "Simply fine-tuning the model not to be sycophantic reduces the rate of successful reward-tampering by up to a factor of 10."
- HHH preference-model reward and HHH prompts in the mix do not prevent generalization (§1 item 4; §5.4).
- Seed variance (§5.4, App. G): "one run overwrites the reward function 2.4x more and overwrites the unit tests 6.7x more than the other. This suggests that many of our observed [differences] in the rates of reward tampering could simply be noise."
- Order versus diversity (§5.4): PPO on the environments sequentially and in parallel gives "nearly identical generalization … These results suggest that it is the diversity of the gameable tasks, not the order of the curriculum, that causes generalization to reward-tampering."

## Scope and limits stated by the authors (§1, §7)
- "None of our models reward-tamper more than 1% of the time, and none evade detection more than 1 in 1,000 trials, even after training on a curriculum where 50% of the environments contain exploitable reward processes."
- "Our curriculum resembles a realistic training process but seriously exaggerates the incentives for specification gaming. We do not find evidence that current frontier models already exhibit sophisticated reward-tampering."
- The authors ran five expert-iteration training sequences in total (§5.1).

## Verification
- Checked on 2026-09-15 against the arXiv PDF text of 2406.10162v3: Abstract, §1-§5, §7, Figures 1, 2, 6, 7, 8.
- Not reported by the source: model size beyond "Claude-2 scale"; expert-iteration and PPO learning rates and batch sizes; per-environment gaming rates in numeric form (figure only).
