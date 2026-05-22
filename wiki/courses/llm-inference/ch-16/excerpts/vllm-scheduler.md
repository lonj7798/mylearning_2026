---
chapter: ch-16
course: llm-inference
phase: read
excerpt_of: "vLLM V1 Scheduler: vllm/v1/core/sched/scheduler.py"
source_url: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
created_at: "2026-05-21"
---

# Excerpt: vLLM V1 Scheduler

**Source files:**
- `vllm/v1/core/sched/scheduler.py` (main loop)
- `vllm/v1/core/sched/output.py` (SchedulerOutput dataclass)
- `vllm/v1/core/sched/utils.py` (token-budget helpers)

**Raw-data source:** [[raw-data/vllm-scheduler]]

---

## The Scheduler class

```python
class Scheduler:
    def __init__(self, vllm_config, kv_cache_config, structured_output_manager):
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_manager = KVCacheManager(kv_cache_config, ...)
        self.structured_output_manager = structured_output_manager

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # SWAPPED is implicit: requests with no allocated blocks but state==WAITING

        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
```

Three queues collapsed into two structures plus an implicit SWAPPED state (preempted requests go back to `waiting` with their progress intact).

---

## The schedule() loop (simplified)

```python
def schedule(self) -> SchedulerOutput:
    scheduled_running = []
    scheduled_new = []
    preempted_req_ids = []
    finished_req_ids = []

    token_budget = self.max_num_scheduled_tokens

    # Step 1: Service running requests (decode tokens, one or more per request).
    self.running.sort(key=lambda r: r.priority, reverse=True)  # priority-aware
    for req in list(self.running):
        num_new_tokens = self._get_num_decode_tokens(req)  # 1 for vanilla decode,
                                                          # K+1 for spec-dec verification
        if token_budget < num_new_tokens:
            break

        if not self.kv_cache_manager.has_capacity_for(req, num_new_tokens):
            # Cache pressure: preempt this request (or a lower-priority one)
            self._preempt(req)
            preempted_req_ids.append(req.request_id)
            continue

        blocks = self.kv_cache_manager.allocate_slots(req, num_new_tokens)
        scheduled_running.append(ScheduledRunning(req, num_new_tokens, blocks))
        token_budget -= num_new_tokens

    # Step 2: Admit waiting requests if budget allows.
    while self.waiting and token_budget > 0 and len(scheduled_running) + len(scheduled_new) < self.max_num_running_reqs:
        req = self.waiting[0]

        # APC prefix-cache lookup
        computed_blocks = self.kv_cache_manager.get_computed_blocks(req)
        num_skipped = len(computed_blocks) * self.cache_config.block_size

        remaining_prompt = req.num_prompt_tokens - num_skipped
        num_new_tokens = min(remaining_prompt, token_budget, self.scheduler_config.max_num_batched_tokens)

        if not self.kv_cache_manager.has_capacity_for(req, num_new_tokens):
            break  # cache pressure → wait

        blocks = self.kv_cache_manager.allocate_slots(req, num_new_tokens, computed_blocks=computed_blocks)
        scheduled_new.append(ScheduledNew(req, num_new_tokens, blocks, computed_blocks))
        token_budget -= num_new_tokens

        if num_new_tokens == remaining_prompt:
            self.waiting.popleft()
            self.running.append(req)
        else:
            # Chunked prefill: request stays in waiting with progress marker
            req.num_computed_tokens += num_new_tokens

    # Step 3: Build structured-output grammar bitmasks for each constrained request.
    grammar_bitmasks = self.structured_output_manager.compute_bitmasks(scheduled_running + scheduled_new)

    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new,
        scheduled_running_reqs=scheduled_running,
        num_scheduled_tokens={r.request_id: n for r, n, _ in scheduled_running + scheduled_new},
        preempted_req_ids=preempted_req_ids,
        finished_req_ids=finished_req_ids,
        grammar_bitmasks=grammar_bitmasks,
        ...
    )
```

The order — running first, then waiting — is the *liveness invariant*. If you reversed it (waiting first), you'd starve in-progress requests under load.

---

## SchedulerOutput dataclass

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_running_reqs: list[RunningRequestData]
    num_scheduled_tokens: dict[str, int]   # request_id → #tokens
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: int  # for prefix-cache aware attention
    finished_req_ids: set[str]
    free_encoder_input_ids: list[tuple[str, int]]
    structured_output_request_ids: dict[str, int]
    grammar_bitmask: torch.Tensor | None
```

This dataclass is the **contract** between the scheduler and the executor/worker. Anything the model needs to know about the step is in here. Anything the scheduler needs to remember is in `self.waiting / self.running / self.kv_cache_manager`.

---

## Preemption modes

```python
def _preempt(self, req: Request):
    if self.scheduler_config.preemption_mode == "swap":
        # Move blocks to CPU memory
        cpu_blocks = self.kv_cache_manager.swap_out(req)
        req.cpu_block_ids = cpu_blocks
    else:  # "recompute"
        # Free all GPU blocks; on resume, recompute the prompt
        self.kv_cache_manager.free(req)
        req.num_computed_tokens = 0  # reset progress

    req.state = RequestStatus.PREEMPTED
    self.waiting.appendleft(req)  # high-priority for resume
    self.running.remove(req)
```

Recompute is the default. Swap requires `--swap-space N` (GB per rank); larger N → more CPU memory headroom but more PCIe traffic on swap-in/out.

---

## Where chunked prefill happens

Chunked prefill is *implicit* in the `min(remaining_prompt, token_budget)` clause in step 2. If the prompt is longer than the budget, only a chunk is scheduled; the request stays in WAITING with updated `num_computed_tokens`.

The attention kernel handles the chunked-prefill case via the `attn_metadata.prefill_mode` flag (FlashAttention varlen or FlashInfer chunked).

---

## Speculative decoding hook

For requests with spec-dec enabled, `_get_num_decode_tokens(req)` returns `K + 1` (K draft + 1 verification anchor) instead of 1. The verification logic runs in the model runner, not the scheduler — the scheduler just budgets the right number of token slots.

```python
def _get_num_decode_tokens(self, req: Request) -> int:
    if req.use_spec_decode:
        return self.spec_decode_config.num_speculative_tokens + 1
    return 1
```

---

## Pitfalls in the source

- **`num_common_prefix_blocks`** is computed only when *all* RUNNING requests share a common prefix (APC's "all-prefix" optimization). This is a serving-stack-specific optimization: shared system prompts across a batch can be attention-computed once. Don't expect this to fire often unless you serve the same system prompt to every request.
- **Cancel handling**: cancelled requests are removed from `waiting` or `running` on the next schedule() call; their blocks are freed.
- **Logger ratelimiting**: V1 scheduler logs a "preempted" warning every N preemptions, not every one. Watch the metrics endpoint (`vllm_preempted_count_total`) for the real number.
- **Token budget vs sequence count**: both can bind. If you raise `max-num-batched-tokens` but leave `max-num-seqs=256`, you're often seq-bound at high concurrency.

---

## Connections

- [[excerpts/vllm-kv-cache-manager]] — block allocation called from here.
- [[excerpts/vllm-structured-output]] — grammar bitmask integration.
- [[excerpts/vllm-production-knobs]] — the flags that tune the loop above.
- [[ch-16]] — parent chapter.
