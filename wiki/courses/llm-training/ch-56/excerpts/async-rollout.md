---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/async-rollout.md
source_url: https://github.com/OpenRLHF/OpenRLHF/blob/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da/openrlhf/trainer/ppo_trainer_async.py
created_at: "2026-04-23"
revised: "2026-09 (generality revision; rewritten to match the verified card)"
---

# Excerpt: OpenRLHF asynchronous PPO — queue, lock, partial rollout

**Source library:** `wiki/raw-data/llm-training/papers/async-rollout.md`
**Version/commit:** `64c1cc4f30d4c0dab772c8aa1dc11288c425e0da`; `ppo_trainer_async.py` is byte-identical at `main` b117b2b
**Files:** `openrlhf/trainer/ppo_trainer_async.py`, `openrlhf/trainer/ray/vllm_engine.py`,
`openrlhf/cli/train_ppo_ray.py`, `README.md`

---

## What ch-56 takes from this source

1. **Two concurrent Ray actors.** `GenerateSamplesActor` and `TrainingActor` run their `fit` loops under
   `PPOTrainerAsync` (L37–350). `--train.async_enable` selects this trainer (`train_ppo_ray.py` L145–148).
2. **Queue and backpressure.** `rollout_queue = Queue(maxsize=queue_size)` with
   `--train.async_queue_size` default 1, plus `rollout_slots`, a queue pre-filled with the same number of
   tokens that acts as a counting semaphore the generator must take from before generating (L287–297).
   If no samples were produced, the token is returned so the loop cannot deadlock (L154–157).
3. **`VLLMLock`.** A Ray actor wrapping `asyncio.Lock`, held around generation in normal mode so weight
   broadcast and generation do not overlap and each batch is generated with one weight set (L19–34).
4. **Partial rollout.** `--train.partial_rollout_enable` requires async mode (`train_ppo_ray.py`
   L277–283). It drops the lock and calls vLLM `pause_generation(mode="keep")` before the broadcast and
   `resume_generation` after (L255–265; `vllm_engine.py` L132–136), so an in-flight sample may contain
   tokens from two weight sets.
5. **No staleness bound.** The enqueued payload has no policy-version field and neither loop filters
   samples by age (L149–151, L227–232). The README states that a larger `async_queue_size` is
   "more off-policy" (L667).
6. **Mode constraints.** Async cannot be combined with `--vllm.enable_sleep` (L666–667);
   `--rollout.vllm_generate_batch_size > --rollout.batch_size` requires async mode (L686–688).

---

## Limits — what this source does not report

The repository reports no throughput or convergence measurement for async mode, and none of the six arXiv
versions of the OpenRLHF paper (2405.11143) contains an async-versus-sync result. The previous version of
this excerpt attributed "1.9× at 7B, 1.6× at 70B" to the paper's §3.3 and Figure 5; §3.3 in v2–v4 is
"PPO Implementation Tricks" and Figure 5 is PPO training curves, so that number has been removed. README
statements that synchronous mode has "better stability" and partial rollout is "most aggressive
off-policy" (L671–679, L731–732) have no experiment behind them.

---

## Links

[[async-rollout]] · [[openrlhf-ppo]] · [[verl-rollout]] · [[on-off-policy-rlhf]]
