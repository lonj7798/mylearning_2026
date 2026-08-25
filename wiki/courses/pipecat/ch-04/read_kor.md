---
title: "worker.run()을 호출하면 무엇이 실행되는가: Queue, Task, 그리고 Out-of-Band Priority"
chapter: ch-04
phase: read
course: pipecat
kind: korean-companion
source: [[read]]
sources:
  - theory-out-of-band-priority
  - pipeline-task-runner
  - frame-processor
  - canonical-voice-bot
  - deployment-scaling
figure: figures/one-call-runtime.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
---

# `worker.run()`을 호출하면 무엇이 실행되는가

> 이 문서는 [[read]] (`ch-04/read.md`)의 한국어 companion입니다. 섹션 번호가 원문과 1:1로 대응하므로
> 나란히 놓고 읽을 수 있습니다. CS/ML 용어는 영어 그대로 둡니다 — frame, processor, pipeline, queue,
> back-pressure, out-of-band, task, worker, runner.

## 왜 이 챕터인가

[[ch-01/read]]은 composition law를 줬습니다 — `link()`는 pointer assignment 두 번이고, `Pipeline._link_processors`는 fold이며, 아무것도 validate하지 않습니다. [[ch-02/read]]는 narrow waist를 줬습니다 — `Frame`은 모두가 합의하는 단 하나의 type이고, 그 대가는 늘어나기만 하는 sum type입니다. [[ch-03/read]]은 여러분이 이미 출시한 `realtime_voice`를 특징지었습니다: 561줄짜리 `VoiceSession` 하나가 supervise하는, 여섯 개의 frozen dataclass로 닫힌 union.

이 세 chapter는 전부 **정지 상태의 structure**에 대한 이야기입니다. 어느 것도 화요일 14:03:22에 한국인 고객이 전화를 받아 "여보세요"라고 말하고, Lina의 오프닝 11초를 듣다가, 네 번째 문장 중간에 끊어버릴 때 무슨 일이 일어나는지는 말해주지 않습니다.

그게 이 chapter입니다. 의도적으로 API 투어가 아닙니다. running example은 정확히 하나 — **Lina의 sales call 한 통** — 이고, chapter에 등장하는 모든 constant, method, queue는 그 통화가 제기하는 질문에 답함으로써 자기 자리를 얻습니다:

- 고객이 문장 중간에 끊는다. 무엇이 tear down되고, 얼마나 빠르며, 이미 output queue에 앉아 있는 3초짜리 한국어 TTS audio는 어떻게 되는가?
- bot이 마지막 멘트를 끝내고 통화를 깔끔하게 종료하고 싶다. 왜 *그* 대기는 unbounded인데, hang-up 대기는 20초로 capped되어 있는가?
- 고객이 4분 동안 아무 말도 안 한다. 아무 일도 일어나지 않는다. 왜? 그리고 그게 sales dial에서 원하는 동작인가?
- STT provider의 websocket handshake가 connect 시점에 멈춘다. 통화는 반쯤 연결된 상태로 그냥 시작되는가?

끝날 때쯤 여러분은 이름 붙은 deliverable 하나를 갖게 됩니다: **Lina host를 위한 process / session / worker topology**. 이후 chapter들이 이걸 소비합니다. [[ch-05/read]]는 여기에 transport를 꽂고, [[ch-10/read]]는 flow-manager frame을 주입하고, [[ch-12/read]]는 여러분의 rule layer를 안에 넣습니다. 어느 것도 이걸 다시 유도하지 않습니다.

숫자를 읽는 방법에 대한 노트. 아래의 모든 Pipecat 줄 번호는 commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`에서 연 것입니다. excerpt library의 내용이 tree와 어긋나는 경우, 조용히 하나를 고르지 않고 본문에서 그렇다고 말합니다. `boson-agent`와 `realtime_voice`에 관한 모든 것은 excerpt 파일([[boson-gateway-server]], [[rtv-pipeline-session]], [[boson-interrupt-subsystem]])에서 온 것이고, 여기가 아니라 거기서 attest됩니다.

---

## 1. 통화 하나, 네 개의 exit

chapter 전체가 이 표 하나에 들어 있습니다. 지금 읽으세요; 아직 이해되지 않을 겁니다. 끝에서 다시 오면 당연하게 읽혀야 합니다.

| 고객이 하는 행동 | exit를 시작하는 frame | 대기가 bounded되는 지점 | constant |
|---|---|---|---|
| Setup: STT/TTS provider connect | (아직 없음 — push된 frame이 하나도 없음) | `_setup_within_timeout`, `worker.py:1104-1121` | `SETUP_TIMEOUT_SECS = 20.0` |
| Start: `StartFrame`이 chain 전체를 통과해야 함 | `StartFrame` | `_wait_for_pipeline_start`, `worker.py:1039-1061` | `START_TIMEOUT_SECS = 20.0` |
| 문장 중간에 끊음 | `CancelFrame` | `_wait_for_pipeline_end`, `worker.py:1063-1095` | `CANCEL_TIMEOUT_SECS = 20.0` |
| bot이 마지막 멘트를 끝냄 | `EndFrame` | **의도적으로 unbounded** | — |
| 침묵함 | (frame의 부재) | `_idle_monitor_handler`, `worker.py:1401-1415` | `IDLE_TIMEOUT_SECS = 300` |

여섯 개의 constant는 전부 worker module 최상단의 한 블록에 선언되어 있습니다. config 파일에 숨겨진 건 없습니다:

**`src/pipecat/pipeline/worker.py:91-100`**
```python
HEARTBEAT_SECS = 1.0
HEARTBEAT_MONITOR_SECS = 10.0

IDLE_TIMEOUT_SECS = 300

CANCEL_TIMEOUT_SECS = 20.0

SETUP_TIMEOUT_SECS = 20.0

START_TIMEOUT_SECS = 20.0
```

이 블록에 대해 즉시 짚고 넘어갈 두 가지가 있습니다.

첫째, `IDLE_TIMEOUT_SECS = 300`은 **5분**입니다. 한국 outbound 보험 dial에서 30초 침묵은 이미 놓친 고객이고, 60초는 수화기를 테이블에 내려놓은 고객입니다. framework의 default는 여러분의 use case 기준으로 order of magnitude만큼 틀렸습니다. 이건 bug가 아닙니다 — Pipecat의 default는 방치된 browser tab이 GPU를 영원히 태우지 않게 하려는 browser demo 기준으로 튜닝되어 있습니다. 이건 여러분이 반드시 재튜닝해야 한다는 뜻이고, §9가 정확한 kwarg를 보여줍니다.

둘째, `END_TIMEOUT_SECS`는 없습니다. graceful path에는 timeout이 아예 없고, §8에서 그 이유를 그대로 말하는 source의 comment를 보게 됩니다. 이 비대칭을 지금 외우세요: **violent path는 bounded, graceful path는 unbounded.** 거의 모두가 반대로 짐작합니다.

> 💡 **쉬운 설명 — 왜 반대로 짐작하게 되나**
> 일반적인 software에서는 "graceful shutdown은 timeout을 걸고, kill은 즉시"가 관용구입니다 (예: Kubernetes의
> `terminationGracePeriodSeconds`). Pipecat이 뒤집는 이유는 도메인이 realtime audio이기 때문입니다.
> graceful path(`EndFrame`)의 **존재 이유가** 큐에 남은 audio를 끝까지 재생하는 것이라, 여기에 timeout을 걸면
> 목적 자체가 사라집니다. 반면 violent path(`CancelFrame`)는 고객이 이미 없어서 지켜야 할 게 없으므로,
> 매달리는 것보다 포기하는 게 낫습니다.

---

## 2. 먼저 바로잡는 vocabulary

무엇보다 먼저: 웹에서 찾게 될 대부분의 Pipecat 자료 — blog 글, YouTube walkthrough, Stack Overflow 답변의 다수, 그리고 LLM이 생성한 Pipecat code의 상당 부분 — 는 `PipelineTask`와 `PipelineRunner`를 기준으로 쓰여 있습니다. 그 이름들은 여전히 동작합니다. 동시에 둘 다 deprecated이고, 그것들로 작성된 code는 여러분이 스스로 떠안기로 선택한 2.0.0 migration입니다.

진짜 이름은 `PipelineWorker`와 `WorkerRunner`입니다.

**`src/pipecat/pipeline/worker.py:1478-1482`**
```python
@deprecated(
    "`PipelineTask` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `PipelineWorker` instead."
)
class PipelineTask(PipelineWorker):
```

**`src/pipecat/pipeline/worker.py:1493-1498`**
```python
@deprecated(
    "`PipelineTaskParams` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `WorkerParams` instead."
)
@dataclass
class PipelineTaskParams(WorkerParams):
```

`src/pipecat/pipeline/runner.py`는 전체가 37줄이고, 그중 마지막 14줄이 이것입니다:

**`src/pipecat/pipeline/runner.py` (tail)**
```python
    "`PipelineRunner` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `WorkerRunner` instead."
)
class PipelineRunner(WorkerRunner):
    """Deprecated alias for :class:`~pipecat.workers.runner.WorkerRunner`.

    .. deprecated:: 1.3.0
        Use :class:`~pipecat.workers.runner.WorkerRunner` instead.
        Will be removed in 2.0.0. The :class:`PipelineRunner` now runs workers
        (of which :class:`~pipecat.pipeline.worker.PipelineWorker` is one kind),
        not just pipelines.
    """

    pass
```

`src/pipecat/pipeline/task.py`는 29줄이고 logic이 전혀 없습니다 — `pipecat.pipeline.worker`에서 다섯 개의 이름을 순수하게 re-export할 뿐입니다:

**`src/pipecat/pipeline/task.py:15-21`**
```python
from pipecat.pipeline.worker import (
    IdleFrameObserver,
    PipelineParams,
    PipelineTask,
    PipelineTaskParams,
    PipelineWorker,
)
```

이 rename은 화장이 아니고 문서화되지 않은 것도 아닙니다. `AGENTS.md`가 한 문장으로 이유를 밝힙니다:

**`AGENTS.md:82`**
```
Terminology note: a "worker" is a runnable unit, "task" now refers only to
asyncio tasks, and cross-worker RPC uses "jobs" and "job groups".
```

이건 특히 여러분에게 중요한데, 이 chapter가 *바로* asyncio task에 관한 것이기 때문입니다. "task"가 여전히 "runnable pipeline unit"을 뜻했다면 아래 문장의 절반이 애매해집니다 — "the process task is cancelled but the task keeps running"은 읽을 수 없는 문장입니다. rename 이후로는 정확합니다: **worker**는 session-scoped runnable이고, **task**는 `asyncio.Task`이며, processor당 그런 task가 두 개 있고 여러분은 곧 둘 다 만나게 됩니다.

역할 분담, 더 이상 궁금해하지 않도록 한 번만 명시합니다:

- **`PipelineWorker(BaseWorker)`** (`worker.py:198`)는 **lifecycle**을 소유합니다 — setup, start, end, cancel, idle detection, heartbeat. 통화당 하나.
- **`WorkerRunner(BaseObject, BusSubscriber)`** (`workers/runner.py:83`)는 **process 관심사**를 소유합니다 — bus, registry, SIGINT/SIGTERM, 그리고 *process*가 언제 멈춰야 하는가라는 질문. host process당 하나.

두 번째 줄이 여러분의 deployment shape를 결정하므로, 별도의 섹션을 갖습니다.

---

## 3. host: `WorkerRunner`, 그리고 Lina에게 `auto_end=False`가 필수인 이유

Lina는 demo bot이 아닙니다. Lina는 FastAPI process로서 하루 종일 앉아 있으면서 여러 고객의 websocket connection을 받고, 오후 마지막 통화가 끊긴 18:00에도 여전히 살아 있어야 합니다.

`WorkerRunner`는 entry point가 정확히 두 개이고, class docstring이 직접 이름을 댑니다:

**`src/pipecat/workers/runner.py:91-100`**
```
    Two entry points:

    - :meth:`add_workers(*workers)` — register one or more workers on the
      runner's bus and start them in the background. Workers run
      concurrently and remaining workers are cancelled when the runner
      ends.
    - :meth:`run` — block until the runner ends. By default
      (``auto_end=True``) the runner ends once every root worker has
      finished; pass ``auto_end=False`` to keep the runner up until
      :meth:`end` / :meth:`cancel` is called.
```

`run()`의 docstring은 Lina의 케이스를 이름으로 짚습니다. framework mechanic에서 제가 추론하는 게 아닙니다 — framework가 직접 말합니다:

**`src/pipecat/workers/runner.py:245-266`**
```
        By default (``auto_end=True``), the runner ends once every root
        worker has finished — so a single-pipeline bot naturally ends
        when its pipeline does. Multi-worker bots whose helpers run
        forever (e.g. waiting for bus messages) end by calling
        :meth:`end` / :meth:`cancel` from an event handler (typically on
        transport disconnect). For long-lived hosts that add and remove
        workers over many sessions (e.g. a FastAPI server), pass
        ``auto_end=False`` so the runner does not exit when no workers
        are left.

        Args:
            worker: Optional worker to run.

                .. deprecated:: 1.3.0
                    Register the worker with :meth:`add_workers` before
                    calling ``run()`` instead.
                    Will be removed in 2.0.0.

            auto_end: When ``True`` (the default), the runner ends once
                every root worker has finished. When ``False``, the
                runner blocks until :meth:`end` or :meth:`cancel` is
                called.
```

"For long-lived hosts that add and remove workers over many sessions (e.g. a FastAPI server)"는 Lina host에 대한 문자 그대로의 서술입니다. default를 그대로 두면 정확히 어떤 일이 벌어지는지, 실제 code로:

**`src/pipecat/workers/runner.py:441-459`**
```python
    async def _run_worker(self, worker: BaseWorker) -> None:
        """Drive a registered worker to completion."""
        try:
            params = WorkerParams(task_manager=self.task_manager)
            await worker.run(params)
        except asyncio.CancelledError:
            pass
        finally:
            # End the runner once every root worker has finished. The
            # current worker's task is still "running" (we're inside its
            # body), so exclude it from the check.
            if self._auto_end and worker.parent is None:
                others_running = any(
                    e.runner_task is not None and not e.runner_task.done()
                    for e in self._entries.values()
                    if e.worker.parent is None and e.worker is not worker
                )
                if not others_running:
                    self._shutdown_event.set()
```

실제 아침 시간에 대입해 trace해 봅시다. 09:14, 고객 한 명이 통화 중이고 `PipelineWorker` 하나가 running입니다. 고객이 끊습니다. `_run_worker`의 `finally`가 발화합니다. root worker가 정확히 하나이고 그게 지금 끝나는 그 worker이므로 `others_running`은 `False`입니다. `self._shutdown_event.set()`. `run()`으로 돌아가서 `await self._shutdown_event.wait()`가 반환되고, runner는 남은 것들을 cancel하고, `cleanup()`을 호출하고, bus를 멈추고, 반환합니다. **고객 한 명이 끊었다는 이유로 여러분의 FastAPI host가 09:14에 종료된 것입니다.** 그날의 이후 모든 통화는 죽은 port에 부딪힙니다.

그게 `:250-253`의 docstring이 경고하는 실패이고, 한 단어짜리 수정입니다. chapter figure의 상단 스트립에서 직접 벌어지는 걸 볼 수 있습니다: `auto_end=False`로 가상 고객을 connect/disconnect 하면 worker가 오가는 동안에도 runner는 살아 있고, `True`로 뒤집으면 마지막 통화가 끝나는 순간 host 전체가 사라집니다.

→ **[one-call-runtime.html](figures/one-call-runtime.html)** — 지금 열어서 §3부터 §9까지 옆에 두고 조작하세요. 세 가지 실험을 순서대로 하세요: 상단 스트립에서 `auto_end`를 뒤집고, 가운데 패널에서 `SystemFrame`을 주입해 두 번 추월하는 걸 보고, 하단 패널에서 네 개의 exit 버튼을 전부 눌러 공유 clock 위에서 `CancelFrame` trace와 `EndFrame` trace를 비교하세요. §7–§8의 bounded/unbounded 비대칭은 그 패널에서는 그림이고 여기서는 문장입니다; 그림을 먼저 보세요.

### 3.1 `add_workers`가 실제로 하는 일, 그리고 없는 것

**`src/pipecat/workers/runner.py:218-235`**
```python
        for worker in workers:
            if worker.name in self._entries:
                logger.error(
                    f"WorkerRunner '{self}': worker '{worker.name}' already exists, skipping"
                )
                continue
            # ``attach`` is async because it also subscribes the worker
            # to the bus — eager subscription is required so workers
            # added later are listening before earlier workers emit
            # their first messages.
            await worker.attach(registry=self._registry, bus=self._bus, worker_runner=self)
            await self._registry.watch(worker.name, self._on_local_worker_ready)
            entry = _WorkerEntry(worker=worker)
            self._entries[worker.name] = entry
            logger.debug(f"WorkerRunner '{self}': added worker '{worker.name}'")

            if self._running:
                await self._start_worker(entry)
```

§13의 deliverable에 필요한 사실 세 가지, 그중 하나는 나쁜 소식입니다.

**(a) 이름은 unique해야 하고, 충돌은 log 한 줄만 남기는 silent no-op입니다.** `worker.name`이 이미 `self._entries`에 있으면 그 worker는 *skip*됩니다 — 시작되지 않고, raise되지도 않습니다. 반복될 수 있는 무언가(고객 전화번호, 재발신되는 CRM lead id)로 worker 이름을 지으면, 같은 번호로의 재발신은 조용히 시작조차 안 됩니다. default 이름은 안전합니다: `BaseObject.__init__`이 `self._name = name or f"{self.__class__.__name__}#{obj_count(self)}"` (`utils/base_object.py:72`)를 하고, `obj_count`는 lock 아래의 per-class `itertools.count`(`utils/utils.py:33-42`)이므로, 이름 없는 worker는 `PipelineWorker#0`, `#1`, `#2` … 를 받습니다. 읽기 좋은 이름을 *원한다면* per-connection session UUID를 쓰세요, 절대 전화번호는 안 됩니다.

**(b) running 중에 추가하면 즉시 시작됩니다.** `if self._running: await self._start_worker(entry)`. 그래서 FastAPI websocket route는 startup 이후 아무 때나 `PipelineWorker`를 생성하고 `await runner.add_workers(w)` 하면 그냥 돌아갑니다. 이게 long-lived runner 하나를 성립시키는 mechanic입니다.

**(c) `remove_workers`가 없습니다.** grep 해봤습니다:

```
$ grep -rn "remove_worker" src/ examples/
(no output)
```

`src/`와 `examples/` 전체에서 0건. `self._entries`는 늘어나기만 하는 `dict[str, _WorkerEntry]`입니다. 끝난 worker의 entry도 그대로 남습니다 — `_run_worker`의 `finally` 블록은 `entry.runner_task.done()`을 확인할 뿐 entry를 삭제하지 않습니다. Lina 통화가 하루치 쌓이면 그 dict는 통화당 entry 하나씩 축적되고, 각 entry는 `PipelineWorker` 참조를 쥐고, 그건 다시 `Pipeline` 전체를 쥐고, 그건 모든 processor와 모든 닫힌 provider client를 쥡니다.

[[pipeline-task-runner]]의 excerpt는 topology를 "`add_workers`로 추가되고 disconnect 시 제거되는 통화당 `PipelineWorker` 하나"라고 서술합니다. 앞 절반은 맞습니다; **뒤 절반은 이 commit에 뒷받침하는 API가 없습니다.** 매끄럽게 넘기지 않고 그대로 말씀드립니다: 하루 장(場) 전체 동안 long-lived `WorkerRunner` 하나를 돌리면 죽은 worker entry가 축적되고, 그건 여러분이 직접 처리해야 합니다 — `runner._entries`에 손을 넣거나(private이고 깨질 겁니다), runner를 pooling하거나, 스케줄에 따라 process를 recycle하거나. §13이 결론을 냅니다.

### 3.2 runner의 shutdown 동사들

method 세 개, 전부 같은 `_shutdown_event`에 대해 idempotent합니다:

**`src/pipecat/workers/runner.py:332-366`**
```python
    async def end(self, reason: str | None = None) -> None:
        """Gracefully end all running workers.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for ending.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"WorkerRunner '{self}': ending gracefully (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.worker.parent is None:
                await self._bus.send(
                    BusEndWorkerMessage(source=self.name, target=name, reason=reason)
                )

    async def cancel(self, reason: str | None = None) -> None:
        """Immediately cancel all running workers.

        Idempotent; subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for cancelling.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"WorkerRunner '{self}': cancelling (reason={reason})")
        self._shutdown_event.set()
        for name, entry in self._entries.items():
            if entry.worker.parent is None:
                await self._bus.send(
                    BusCancelWorkerMessage(source=self.name, target=name, reason=reason)
                )
```

`for` loop를 주의 깊게 읽으세요: **이 둘 다 runner 위의 모든 root worker를 때립니다.** `runner.cancel()`은 "이 통화를 cancel"이 아닙니다. "이 host에 현재 있는 모든 통화를 cancel"입니다.

repo의 canonical example이 정확히 그걸 하고, 그 example은 통화당 process 하나이므로 *거기서는* 맞습니다:

**`examples/getting-started/06-voice-agent.py:116-119`**
```python
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()
```

저 줄을 multi-tenant Lina host에 복사하면 고객 한 명이 끊을 때 그 박스의 다른 모든 고객이 떨어집니다. per-call 동사는 `runner.cancel()`이 아니라 `worker.cancel()`입니다. 이건 이 framework의 example set 전체에서 가장 파급력 큰 copy-paste 위험이고, `runner.py:350-366`을 읽지 않으면 보이지 않습니다.

signal은 runner 전체 경로로 라우팅되는데, container SIGTERM에는 그게 여러분이 원하는 바입니다:

**`src/pipecat/workers/runner.py:529-537`**
```python
    def _sig_handler(self) -> None:
        if not self._sig_task:
            self._sig_task = asyncio.create_task(self._sig_cancel())

    async def _sig_cancel(self) -> None:
```

handler는 `:515` (`loop.add_signal_handler(signal.SIGINT, ...)`)과 `:524` (SIGTERM)에 설치되고, constructor(`:114-115`)의 `handle_sigint=True` / `handle_sigterm=False`로 gate됩니다. default를 보세요: **SIGINT는 yes, SIGTERM은 no.** Kubernetes와 대부분의 container runtime은 eviction 시 SIGTERM을 보냅니다. Lina를 container에 배포하면서 `handle_sigterm=False`를 그대로 두면, rolling restart가 graceful `BusEndWorkerMessage` 경로를 전혀 거치지 않고 process를 죽입니다. `handle_sigterm=True`로 설정하세요.

---

## 4. processor 하나의 내부: queue 둘, task 둘

이제 host에서 processor 하나까지 끝까지 zoom in 합니다. 이게 이후 모든 chapter를 읽을 수 있게 만드는 mechanic입니다 — [[ch-06/read]]의 turn boundary, [[ch-08/read]]의 barge-in cascade, [[ch-12/read]]의 rule layer는 전부 "frame이 도착한다"와 "여러분의 `process_frame`이 실행된다" 사이에 정확히 무슨 일이 일어나는지 아는 데 달려 있습니다.

한 문장 버전: **모든 `FrameProcessor`는 두 개의 asyncio queue를 소유하고 두 개의 asyncio task가 그것을 drain하며, `SystemFrame`은 오직 각각의 첫 번째만 건드립니다.**

### 4.1 priority queue

**`src/pipecat/processors/frame_processor.py:132-171`**
```python
class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for the frames arriving at a frame processor.

    Frames are dequeued in three tiers: the `StartFrame` first, then
    `SystemFrame`, then data and control frames. Frames of the same tier keep
    their arrival order.

    """

    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()
        # Counts every frame enqueued, which keeps frames of the same tier in
        # arrival order and stops the queue from ever having to compare frames.
        self.__counter = 0

    async def put(self, item: tuple[Frame, FrameDirection, FrameCallback | None]):
        """Put an item into the priority queue.

        The `StartFrame` outranks every other frame and `SystemFrame` frames
        outrank data and control frames.

        Args:
            item: The frame to enqueue, with its direction and callback.

        """
        frame, _, _ = item
        if isinstance(frame, StartFrame):
            priority = self.START_PRIORITY
        elif isinstance(frame, SystemFrame):
            priority = self.SYSTEM_PRIORITY
        else:
            priority = self.DEFAULT_PRIORITY

        self.__counter += 1
        await super().put((priority, self.__counter, item))
```

tier 셋, `isinstance`로 배정, 설정 가능한 것 없음. `__counter`는 동시에 두 가지 일을 하고 comment가 둘 다 이름을 댑니다: tier **내에서** FIFO를 보존하고, "stops the queue from ever having to compare frames". 그 두 번째 역할이 load-bearing입니다. `asyncio.PriorityQueue`는 `heapq`를 쓰고, heapq는 tuple을 element-wise로 비교합니다; 두 entry의 priority가 같으면 Python은 다음 element로 넘어갑니다. Frame은 ordering이 없는 `@dataclass` instance라, 동점이 나면 `TypeError: '<' not supported between instances of 'TTSAudioRawFrame' and 'TTSAudioRawFrame'`가 발생합니다. monotonic counter가 두 번째 element는 절대 동점이 안 되게 보장하므로, 세 번째 element — frame — 은 결코 비교되지 않습니다. 부하 상황, 즉 같은 tier의 frame 두 개가 동시에 상주할 때만 드러날 crash에 대한 두 줄짜리 방어입니다. 그런데 voice pipeline에서는 그게 항상입니다.

> 💡 **쉬운 설명 — heapq의 tuple 비교**
> `heapq`에 `(10, 3, frameA)`와 `(10, 3, frameB)`를 넣으면 첫 element 10 동점, 두 번째 3 동점 → Python은
> `frameA < frameB`를 시도하고, dataclass에 `order=True`가 없으니 `TypeError`로 터집니다.
> counter가 단조 증가하므로 `(10, 3, ...)`과 `(10, 4, ...)`가 되어 두 번째에서 항상 결판이 나고,
> frame 자체는 비교 대상이 되지 않습니다. 세 개짜리 tuple의 두 번째 자리가 "tie-breaker이자 crash 방어"인 겁니다.

`StartFrame`이 `SystemFrame` 위에 자기 tier를 갖는 건 `queue_frame`을 읽기 전까지는 over-engineering처럼 보입니다:

**`src/pipecat/processors/frame_processor.py:700-728`**
```python
    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: FrameCallback | None = None,
    ):
        """Queue a frame for processing.

        Args:
            frame: The frame to queue.
            direction: The direction of frame flow.
            callback: Optional callback to call after processing.
        """
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        if self._enable_direct_mode:
            await self.__process_frame(frame, direction, callback)
            return

        await self.__input_queue.put((frame, direction, callback))

        # Nothing drains the queue until the StartFrame arrives, so a processor
        # never acts on a frame before it has been started. Frames pushed
        # between setup and the StartFrame simply wait, and the StartFrame is
        # dequeued ahead of them.
        if isinstance(frame, StartFrame):
            self.__create_input_task()
```

input task는 `StartFrame`이 도착하기 전까지 존재하지 않습니다. 그래서 `setup()`(여러분의 STT service가 websocket을 여는 곳)과 `StartFrame` 사이에는 frame이 *enqueue*는 되지만 아무것도 drain하지 않는 window가 있습니다. provider client가 connect 도중 error frame을 push하면 그건 queue에 앉아 있습니다. 마침내 `StartFrame`이 도착하면 그것이 priority 1로 **가장 먼저** dequeue되어, 이미 기다리고 있던 모든 것보다 앞섭니다. 그게 세 번째 tier의 존재 이유 전부입니다: wall-clock 상 더 일찍 도착한 것에 대해서조차, processor가 무언가에 대해 행동하기 전에 반드시 started 상태임을 보장합니다.

`queue_frame`에 대해 나중에 물기 때문에 지금 챙겨둬야 할 세 가지:

1. **`self._cancelling`은 silent drop입니다.** `CancelFrame` 이후 `queue_frame`은 enqueue도 안 하고 log도 없이 반환합니다. frame이 사라집니다. §7이 `_cancelling`이 설정되는 지점을 보여줍니다.
2. **`_enable_direct_mode`는 두 queue를 모두 건너뜁니다.** §4.4에서 더 다룹니다.
3. **`queue_frame`이 *유일한* entry point입니다.** `push_frame`은 이웃의 `process_frame`을 호출하지 않습니다 — 이웃의 `queue_frame`을 호출합니다. push는 항상 이웃에 enqueue하는 것입니다. 그게 각 hop을 독립적인 scheduling point로 만들고, `link()`가 세 줄일 수 있는 이유입니다.

### 4.2 분기 — 트릭의 전부, 아홉 줄

**`src/pipecat/processors/frame_processor.py:1287-1313`**
```python
    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queue for another task
        to execute.

        """
        while True:
            (frame, direction, callback) = await self.__input_queue.get()

            if self.__should_block_system_frames and self.__input_event:
                logger.trace(f"{self}: system frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_system_frames = False
                logger.trace(f"{self}: system frame processing resumed")

            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
            else:
                raise RuntimeError(
                    f"{self}: __process_queue is None when processing frame {frame.name}"
                )

            self.__input_queue.task_done()
```

`if/elif`를 보세요. `SystemFrame`은 **바로 여기, input task 위에서 실행**됩니다. 나머지는 전부 *중계*됩니다 — 두 번째 queue에 올려놓고 잊어버립니다.

두 번째 queue는 자기 task를 갖습니다:

**`src/pipecat/processors/frame_processor.py:1315-1333`**
```python
    async def __process_frame_task_handler(self):
        """Handle non-system frames from the process queue."""
        while True:
            self.__process_current_frame = None

            (frame, direction, callback) = await self.__process_queue.get()

            self.__process_current_frame = frame

            if self.__should_block_frames and self.__process_event:
                logger.trace(f"{self}: frame processing paused")
                await self.__process_event.wait()
                self.__process_event.clear()
                self.__should_block_frames = False
                logger.trace(f"{self}: frame processing resumed")

            await self.__process_frame(frame, direction, callback)

            self.__process_queue.task_done()
```

이제 추월 횟수를 세어봅시다. 현재 75개의 audio frame이 queue에 쌓인 processor에 도착한 `SystemFrame`은 그것들을 **두 번** 앞지릅니다:

1. **들어올 때, priority로.** `FrameProcessorQueue.put`이 10을 주고, audio frame들은 20입니다. input task는 도착 순서와 무관하게 그것을 다음으로 pop합니다.
2. **느린 queue에 아예 들어가지 않음으로써.** 75개의 audio frame은 `__process_queue`에서 `__process_frame_task_handler`를 기다리는데, 그 handler는 여러분의 TTS handler 안에서 `await` 중이라 바쁩니다. system frame은 그 줄에 서지 않습니다. input task 위에서 실행되고, input task는 pop하고 routing하는 게 유일한 일이라 한가합니다.

그래서 3초짜리 audio가 아직 상주하는 와중에도 `InterruptionFrame`이 모든 processor에 한 자리수 millisecond 안에 도달하는 겁니다. 덧붙여진 fast path가 아닙니다; *다른 task*입니다.

[[theory-out-of-band-priority]]는 같은 선을 2001년에 그것을 그린 GStreamer까지 거슬러 올라갑니다. 거기서 downstream event는 "in-band (serialised with the buffer flow)" 아니면 "out-of-band (travelling through the pipeline instantly … skipping ahead of buffers being processed or queued in the pipeline)" 둘 중 하나입니다. `SEGMENT`, `CAPS`, `TAG`, `EOS`는 serialised이고; `FLUSH_START`는 out-of-band입니다. buffers:events가 DataFrame:SystemFrame에 대응하고, EOS-is-serialised가 `EndFrame`-is-a-`ControlFrame`에 대응합니다 — 그게 §8입니다. Pipecat이 이걸 발명하지 않았습니다; 재유도했고, `src/pipecat/processors/gstreamer/pipeline_source.py:39`(`class GStreamerPipelineSource(FrameProcessor)`)에서 실제 GStreamer graph를 vendoring합니다.

그리고 pause machinery가 2단인 것도 같은 이유입니다. `pause_processing_frames()`는 `__process_event`를 통해 process task를 gate하고; `pause_processing_system_frames()`는 `__input_event`를 통해 input task를 gate합니다. control flow는 살려둔 채 data flow만 얼릴 수 있는데, 이건 예컨대 tool call이 resolve되는 동안 audio를 붙잡아 두면서도 barge-in은 여전히 도달하기를 원할 때 정확히 필요한 것입니다.

### 4.3 cancel 가능한 쪽 절반

이 비대칭에는 latency 이상의 목적이 있습니다: **process task는 버릴 수 있고 input task는 그렇지 않습니다.**

**`src/pipecat/processors/frame_processor.py:1130-1150`**
```python
    async def _start_interruption(self):
        """Start handling an interruption by cancelling current tasks."""
        try:
            current_is_uninterruptible = isinstance(
                self.__process_current_frame, UninterruptibleFrame
            )
            if current_is_uninterruptible:
                # The frame currently being processed is uninterruptible, so we
                # must not cancel it. Just flush non-uninterruptible frames from
                # the queue; any uninterruptible ones will be kept and processed
                # after the current frame finishes.
                self.__reset_process_queue()
            else:
                # Cancel and re-create the process task. Previously this branch
                # was skipped when the queue contained an uninterruptible frame,
                # which caused slow non-uninterruptible frames to block
                # interruptions. Uninterruptible queued frames are safe here
                # because __create_process_task calls __reset_process_queue
                # internally, which always preserves them.
                await self.__cancel_process_task()
                self.__create_process_task()
```

`await self.__cancel_process_task()`는 coroutine을 **`await` 도중에** 죽입니다 — 여러분의 `process_frame` 안 어디에 있든 상관없이. 그다음 새 task와 새 queue. input task는 건드리지 않으므로, data 쪽 절반이 재건되는 동안에도 processor는 여전히 *다음* system frame을 듣고 있습니다.

선택적 flush는 `FrameQueue.reset()`입니다:

**`src/pipecat/utils/frame_queue.py:84-95`**
```python
    def reset(self) -> None:
        """Remove all non-UninterruptibleFrame items, keeping uninterruptible ones."""
        kept: asyncio.Queue = asyncio.Queue()
        while not self.empty():
            item = self.get_nowait()
            if isinstance(self._frame_getter(item), UninterruptibleFrame):
                kept.put_nowait(item)
            self.task_done()
        while not kept.empty():
            item = kept.get_nowait()
            self.put_nowait(item)
            kept.task_done()
```

전부 drain하고, `UninterruptibleFrame`만 다시 enqueue합니다. 반쯤 말해진 한국어 TTS 응답은 증발하고; queue에 있던 `EndFrame`은 살아남습니다, `EndFrame`이 mixin을 달고 있기 때문입니다:

**`src/pipecat/frames/frames.py:1899-1910`**
```python
class EndFrame(ControlFrame, UninterruptibleFrame):
    """Frame indicating pipeline has ended and should shut down.

    Indicates that a pipeline has ended and frame processors and pipelines
    should be shut down. If the transport receives this frame, it will stop
    sending frames to its output channel(s) and close all its threads. Note,
    that this is a control frame, which means it will be received in the order it
    was sent.

    This frame is marked as UninterruptibleFrame to ensure it is not lost when
    an InterruptionFrame is processed. Terminal frames must survive interruption
    to guarantee proper pipeline shutdown.
```

`_uninterruptible_count` 부기는 `_put`/`_get`을 override(`frame_queue.py:72-81`)해서 O(1)로 유지되므로, `has_uninterruptible`은 scan이 아니라 비교입니다.

> **source에 틀린 게 하나 있으니 그대로 말합니다.** `has_queued_frame`의 docstring(`frame_processor.py:1244-1249`)은 그 확인이 "is O(distinct enqueued types) with no queue scanning"이라고 주장합니다. 이 method는 `FrameQueue.has_frame`에 위임하는데, 그 body는 `for item in self._queue: if isinstance(...): return True` (`frame_queue.py:64-67`) — deque에 대한 linear scan입니다. O(1) 주장은 `has_uninterruptible`(counter)에 대해서는 참이고 `has_frame`에 대해서는 거짓입니다. voice-pipeline queue 깊이에서는 문제되지 않지만, 저 docstring 위에 hot loop를 만들지는 마세요.

flush가 *downstream*에서 무슨 일을 하는지 — TTS service의 aggregation buffer, audio clock, 여러분이 손댈 수 없는 Twilio의 playout buffer — 는 [[ch-08/read]]의 주제입니다. 여러분이 이미 output queue를 소유하게 될 그곳에서 한 번만 가르칩니다. 여기서는 이것만 알면 됩니다: **flush는 선택적이고, out-of-band signal로 trigger된다.**

### 4.4 세 번째 경로, 그리고 framework 자신이 그걸 쓰는 곳

`_enable_direct_mode`는 두 queue를 모두 우회합니다. `frame_processor.py`에서 여섯 번 등장하고(`:243, 717, 784, 1206, 1224, 1233`), 모든 사용처가 같은 모양입니다: 즉시 return하거나, inline으로 처리하거나.

**`src/pipecat/processors/frame_processor.py:1222-1229`**
```python
    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if self._enable_direct_mode:
            return

        if not self.__process_frame_task:
            self.__reset_process_task()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())
```

이건 debug switch가 아닙니다. grep에 따르면 framework 자신의 구조적 processor들은 전부 direct mode입니다:

```
$ grep -rn "enable_direct_mode=True" src/pipecat/
src/pipecat/pipeline/service_switcher.py:388
src/pipecat/pipeline/service_switcher.py:395
src/pipecat/pipeline/sync_parallel_pipeline.py:76
src/pipecat/pipeline/sync_parallel_pipeline.py:109
src/pipecat/pipeline/pipeline.py:36
src/pipecat/pipeline/pipeline.py:72
src/pipecat/pipeline/pipeline.py:113
src/pipecat/tests/utils.py:103
```

`pipeline.py:36`은 `PipelineSource`, `:72`는 `PipelineSink`, `:113`은 `Pipeline` 자체입니다:

**`src/pipecat/pipeline/pipeline.py:99-121`**
```python
    def __init__(
        self,
        processors: Sequence[FrameProcessor],
        *,
        source: FrameProcessor | None = None,
        sink: FrameProcessor | None = None,
    ):
        """Initialize the pipeline with a list of processors.

        Args:
            processors: Sequence of frame processors to connect in sequence.
            source: An optional pipeline source processor.
            sink: An optional pipeline sink processor.
        """
        super().__init__(enable_direct_mode=True)

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = source or PipelineSource(self.push_frame, name=f"{self}::Source")
        self._sink = sink or PipelineSink(self.push_frame, name=f"{self}::Sink")
        self._processors: list[FrameProcessor] = [self._source, *processors, self._sink]

        self._link_processors()
```

이건 `Pipeline`이 그 자체로 `FrameProcessor`라는 [[ch-01/read]]의 법칙이 낳는, 진짜로 중요한 귀결입니다. container가 자기 queue와 task를 가졌다면, pipeline 안에 pipeline을 nesting하는 것 — worker가 두 번 하는 일, §6.3 참조 — 은 nesting level마다 queue hop 하나와 task 두 개를 추가했을 겁니다. 대신: **container는 queueing을 0만큼 추가합니다.** `Pipeline.process_frame`은 그냥 `self._source.queue_frame(...)` 또는 `self._sink.queue_frame(...)`으로 routing하고(`pipeline.py:192-195`), source/sink 자신도 direct-mode입니다. 이 system의 모든 queue는 실제 일을 하는 *leaf* processor의 것입니다. composition은 authoring time뿐 아니라 runtime에도 공짜입니다.

> 💡 **쉬운 설명 — "container는 queue를 0 추가한다"가 왜 큰일인가**
> §6.3에서 보게 되듯 worker는 여러분의 `Pipeline([7개])`를 `Pipeline([RTVIProcessor, Pipeline([7개])])`로
> 다시 감쌉니다. 만약 `Pipeline`이 보통의 processor처럼 자기 input queue + process queue와 task 두 개를
> 가졌다면, 이 nesting은 frame 하나당 queue hop을 2~3개 더 태우고 event-loop scheduling 왕복을 그만큼
> 추가했을 겁니다. `enable_direct_mode=True`라서 `Pipeline.process_frame`은 그냥 안쪽 `_source`의
> `queue_frame`을 곧장 호출합니다 — 함수 호출 한 번, queue 0개. 그래서 §4.1에서 센 "audio frame 75개가
> 쌓인 queue"는 항상 STT/TTS/LLM 같은 **leaf** processor의 것이지, container의 것이 아닙니다.
> 즉 *N/r*의 *N*은 nesting 깊이와 무관합니다.

trade-off는 [[frame-processor]]에 명시되어 있고 실제합니다: direct mode에서는 ordering 보장이 적용되지 않습니다, 순서를 매길 queue가 없기 때문입니다. 여러분이 작성한 processor에는 설정하지 마세요.

---

## 5. 여러분의 code가 아무리 좋아도 참인 두 가지

이 chapter가 소유하는 두 개의 이론 문단이고, [[theory-out-of-band-priority]]가 first principle로부터 펼치는 논증입니다. 이 course의 나머지 전부가 이걸 인용합니다; 다른 어디에서도 재유도하지 않습니다.

### 5.1 control latency는 queue depth 나누기 drain rate

먼저 구체적 케이스부터.

Lina가 문장 중간입니다: *"고객님, 이 상품은 65세까지 갱신 없이 보장이 되고요, 지금 가입하시면…"* 한국어 TTS vendor는 그 3초짜리 발화 전체를 websocket으로 약 400 ms 만에 여러분에게 stream했습니다 — vendor는 그렇게 합니다, 빠른 게 그들의 일이고 audio는 network 대역폭보다 작으니까요.

그 audio는 지금 어디 있을까요? output transport의 queue 안입니다:

**`src/pipecat/transports/base_output.py:690`**
```python
                self._audio_queue = FrameQueue()
```

이건 **clock-paced** task인 `_clock_task_handler`(`base_output.py:1079`)가 drain합니다 — 즉 realtime으로 drain됩니다, audio가 재생될 수 있는 rate가 그것뿐이니까요. 그리고 chunk 단위로 write되는데, 그 크기는 여기서 정해집니다:

**`src/pipecat/transports/base_output.py:132-136`**
```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

default는 **`src/pipecat/transports/base_transport.py:72`**에 설정되어 있습니다:
```python
    audio_out_10ms_chunks: int = 4
```

10 ms chunk 네 개 = write되는 chunk당 **40 ms의 PCM**. 그러므로 3초짜리 한국어 audio는 `_audio_queue`에 대략 75개 chunk가 상주하고, 여기에 더해 carrier가 여러분 downstream에서 이미 들고 있는 양이 있습니다.

이제 고객이 barge-in 한다고 해봅시다 — "아니 잠깐만요" — 그런데 여러분의 stop signal이 그 audio 뒤에 줄 서는 평범한 frame이라고 합시다. queue는 realtime으로 drain됩니다. 여러분의 stop signal은 3초짜리 realtime audio 뒤에 있습니다. **3초 늦게 도착합니다.** handler가 느려서가 아닙니다. handler가 `return` 한 줄이어도 여전히 3초 늦게 도착합니다.

이제 벌어지는 걸 봤으니, 공식: 엄격히 ordered된 pipe는 channel이 하나입니다. 시각 *t*에 enqueue된 control message는 *t* 이전에 enqueue된 모든 것이 drain된 뒤에야 전달됩니다. 상주 item이 *N*개이고 sink가 realtime rate *r*로 drain하면:

```
control latency = N / r
```

이고 *r*은 물리로 고정되어 있습니다 — 소리가 speaker를 떠나는 rate입니다. 유일한 자유 변수는 *N*입니다. 이건 code를 최적화해서 좋아지지 않습니다. 그 queue에 있지 않음으로써 좋아집니다.

여기서 떨어져 나오는 것에 주목하세요: `audio_out_10ms_chunks`는 buffering tuning knob이 아니라 **interrupt-granularity** knob이고, source comment가 그렇게 말합니다 — "This will help with interruption handling." chunk가 작을수록 write loop가 cancellation을 더 자주 확인하고, 대가는 syscall이 늘어나는 것입니다. 그게 실제 trade이고, [[ch-11/read]]이 latency budget을 만들 때 쓰게 될 숫자 중 하나입니다.

그 한 문단이 애초에 priority tier가 존재하는 이유입니다. §4의 모든 것은 *N/r*의 귀결입니다.

> 💡 **쉬운 설명 — N/r을 숫자로**
> chunk 하나 = 40 ms. 3초 audio = 75 chunk. drain rate *r*은 초당 25 chunk(40 ms마다 하나)로 물리 고정.
> 그러면 in-band stop signal의 latency = 75 / 25 = **3초**. 만약 `audio_out_10ms_chunks=1`(10 ms chunk)로
> 줄이면 chunk는 300개, drain rate는 초당 100개 → 여전히 3초입니다. **chunk 크기를 줄여도 N/r은 안 변합니다.**
> 줄어드는 건 "현재 쓰고 있는 chunk를 끝내기까지 걸리는 시간"(40 ms → 10 ms), 즉 cancellation 확인 간격입니다.
> 그래서 buffering knob이 아니라 interrupt-granularity knob이라는 겁니다. 3초를 없애는 유일한 방법은
> 그 queue에 아예 서지 않는 것, 즉 out-of-band입니다.

### 5.2 살아 있는 microphone에는 back-pressure를 걸 수 없다

back-pressure는 consumer가 producer에게 "천천히 해"라고 말하는 것입니다. 정본 서술은 Reactive Streams spec(v1.0.4, 2022-05-26)입니다: 목적은 "asynchronous stream processing with non-blocking backpressure"이고, backpressure는 "in order to allow the queues which mediate between threads to be bounded" 존재합니다.

Pipecat의 queue는 bounded되어 있지 않습니다. 여기 grep이 있고, 이게 증거의 전부입니다:

```
$ grep -n maxsize src/pipecat/processors/frame_processor.py \
                  src/pipecat/utils/frame_queue.py \
                  src/pipecat/transports/base_input.py \
                  src/pipecat/transports/base_output.py
$ echo $?
1
```

네 파일 전체에서 0건. `FrameProcessorQueue.__init__`은 맨 `super().__init__()`을 호출하고(`frame_processor.py:146-151`), `FrameQueue.__init__`도 마찬가지이며(`frame_queue.py:43`), inbound audio queue는:

**`src/pipecat/transports/base_input.py:265`**
```python
            self._audio_in_queue = asyncio.Queue()
```

`maxsize` 없음. framework 어디에도 demand signal 없음.

이건 실수가 아니고, 논거는 architectural이 아니라 물리적입니다. **producer를 block해도 speaker가 멈추지는 않습니다.** 여러분의 STT processor가 읽기를 멈추면, microphone은 계속 capture하고, carrier는 계속 보내고, audio는 *다른 어딘가*에 buffering됩니다 — OS socket buffer, transport library, Twilio 서버. 일의 양을 줄인 게 아닙니다; queue를 여러분이 볼 수도 없고 flush할 수도 없는 곳으로 옮긴 것이고, latency는 그 통화가 끝날 때까지 단조 증가합니다.

realtime media pipeline에는 진짜 선택지가 정확히 둘 있습니다: **drop** 아니면 **flush**. Pipecat은 out-of-band signal로 trigger되는 selective flush를 택합니다 — 그게 §4.3의 `_start_interruption` → `FrameQueue.reset()`이고, `UninterruptibleFrame`은 유지합니다.

여러분의 `realtime_voice`는 다른 쪽 선택지를 택했고, 여러분이 이미 production에서 그게 어떻게 동작하는지 알기 때문에 정확히 짚어둘 가치가 있습니다. [[rtv-pipeline-session]]에 따르면 `VoiceSessionConfig`는 모든 queue를 bound합니다 — `ingress_queue_size=64`, `event_queue_size=256`, `phrase_queue_size=8`, `audio_queue_size=32` — 그리고 **세 가지 서로 다른 overflow policy**를 적용합니다: ingress는 reject-on-overflow(`push_audio`가 `QueueOverflowError("ingress queue full; frame rejected instead of adding latency")`를 raise)이고, phrase queue와 audio queue는 backpressure를 겁니다. class docstring이 근거를 명시합니다: *"Ingress uses reject-on-overflow so a transport cannot silently extend user turn latency."*

두 설계를 점수 매기지 말고 나란히 놓으세요, 점수 매기는 건 [[ch-13/read]]의 일이니까:

- **`realtime_voice`**: bounded ingress, overflow 시 **drop**, latency가 조용히 늘어나지 못하게 하려고 선택. drop된 frame은 사라지고; transport는 exception으로 그 사실을 압니다.
- **Pipecat**: unbounded queue, out-of-band signal에 의한 **flush**, barge-in이 pipeline 전체를 한 번에 비울 수 있게 하려고 선택. 무언가가 drop하라고 말하기 전까지는 아무것도 drop되지 않습니다.

둘 다 *N/r*에 대한 답입니다. `realtime_voice`는 admission에서 *N*을 cap합니다. Pipecat은 *N*이 자라게 두고 명령이 오면 비웁니다. 각각이 *무엇을 하는지*는 사실이고; 어느 쪽이 Lina에 맞는지는 아직 여러분에게 없는 증거로 답할 질문입니다.

---

## 6. Exit 하나 — setup과 start

이제 네 개의 exit를 순서대로 걷습니다. 새벽 2시에 터지기 전까지 아무도 생각하지 않는 것부터 시작합니다.

`worker.run(params)`는 lifecycle 전체를 담은 method 하나입니다:

**`src/pipecat/pipeline/worker.py:748-791`**
```python
    async def run(self, params: WorkerParams):
        """Start and manage the pipeline execution until completion or cancellation.

        Args:
            params: Configuration parameters for pipeline execution.
        """
        if self.has_finished():
            return

        try:
            # Setup processors.
            if not await self._setup_within_timeout(params):
                # Nothing was pushed into the pipeline, so there is nothing to
                # drain: release whatever was set up and give up.
                await self._cleanup(cleanup_pipeline=True)
                return

            # Create the worker's tasks and wait for the push task, which
            # feeds frames to the very beginning of our pipeline (i.e. to
            # our controlled source processor).
            await self._create_tasks()

            try:
                # Wait for pipeline to finish.
                await self._wait_for_pipeline_finished()
            except asyncio.CancelledError:
                logger.debug(f"Pipeline worker {self} got cancelled from outside...")
                # We have been cancelled from outside, let's just cancel everything.
                await self._cancel()
                # Wait again for pipeline to finish. This time we have really
                # cancelled, so it should really finish.
                await self._wait_for_pipeline_finished()
                # Re-raise in case there's more cleanup to do.
                raise
        finally:
            # We can reach this point for different reasons:
            #
            # 1. The pipeline worker has finished (try case).
            # 2. By an asyncio worker cancellation (except case).
            logger.debug(f"Pipeline worker {self} is finishing...")
            await self._cancel_tasks()
            self._print_dangling_tasks()
            self._finished = True
            logger.debug(f"Pipeline worker {self} has finished")
```

세 단계: `_setup_within_timeout` → `_create_tasks` → `_wait_for_pipeline_finished`.

`except asyncio.CancelledError` 블록에 주목하세요. worker를 밖에서 `asyncio.cancel` 하면 — 손이 먼저 가는 그 방법 — worker는 그냥 죽지 **않습니다**. 자기 자신의 `_cancel()`을 호출하고, `CancelFrame`이 pipeline을 통과하기를 기다리고, 그러고 나서야 re-raise합니다. 그게 [[pipeline-task-runner]]가 우회하지 말라고 경고하는 cooperative shutdown입니다. 동작은 하지만, asyncio cancellation 아래에 §7의 20초 bound가 겹쳐 깔리는데 이건 debug하기 헷갈리는 상황입니다. shutdown을 `stop_when_done()`이나 `cancel()`로 몰면 이걸 볼 일이 없습니다.

### 6.1 `SETUP_TIMEOUT_SECS` — frame이 없는 exit

**`src/pipecat/pipeline/worker.py:1104-1121`**
```python
    async def _setup_within_timeout(self, params: WorkerParams) -> bool:
        """Set up the pipeline worker and all processors, bounded by a timeout.

        Returns:
            Whether everything was set up. A processor that blocks while being
            set up never lets the pipeline start, so setting up is abandoned
            once ``setup_timeout_secs`` elapses.
        """
        try:
            await asyncio.wait_for(self._setup(params), timeout=self._setup_timeout_secs)
            return True
        except TimeoutError:
            logger.error(
                f"{self}: timeout setting the pipeline up "
                "(a processor blocked while connecting?), stopping the pipeline."
            )
            await self._call_event_handler("on_setup_timeout")
            return False
```

`setup()`은 processor들이 connect하는 곳입니다: STT가 vendor로 websocket을 열고, TTS가 authenticate하고, LLM client가 session을 만듭니다. 한국어 STT provider의 endpoint가 나쁜 1분을 보내고 있어서 connect가 멈추면, 통화가 부딪히는 벽이 이것입니다 — 20초 뒤 `on_setup_timeout`이 발화하고 `run()`은 곧장 `_cleanup(cleanup_pipeline=True)`으로 가서 반환합니다.

`run()`의 comment가 중요한 성질을 짚습니다: *"Nothing was pushed into the pipeline, so there is nothing to drain."* `StartFrame`이 만들어진 적 없고, processor가 start된 적 없고, frame이 queue에 들어간 적 없습니다. 이 exit path는 frame이 단 하나도 pipeline을 통과하지 않은 채 pipeline이 tear down되는 유일한 경로입니다. 그래서 cancel path를 재사용하지 않고 자기 constant가 필요한 겁니다.

Lina 기준: live carrier connection을 아무것도 재생하지 않은 채 열어두기에 20초는 깁니다. telephony provider가 "여보세요" 이후 20초 동안 고객에게 침묵을 stream하고 있다면 그 통화는 이미 진 겁니다. `setup_timeout_secs`는 constructor kwarg(`worker.py:299`)입니다; 5초를 고려하세요.

### 6.2 `START_TIMEOUT_SECS` — `StartFrame`은 끝까지 도달해야 한다

setup이 성공했으니, `_create_tasks()`가 단일 push task(`worker.py:986-989`)를 spawn하고, 그 task가 engine을 돌립니다:

**`src/pipecat/pipeline/worker.py:1205-1246`**
```python
    async def _process_push_queue(self):
        """Process frames from the push queue and send them through the pipeline.

        This is the worker that runs the pipeline for the first time by sending
        a StartFrame and by pushing any other frames queued by the user. It runs
        until the worker is cancelled or stopped (e.g. with an EndFrame).
        """
        self._maybe_start_idle_task()

        # Processors read the pipeline configuration from FrameProcessorSetup,
        # but the deprecated StartFrame fields carry it until they are removed,
        # so that a processor still reading one gets the configured value.
        start_frame = StartFrame(
            audio_in_sample_rate=self._params.audio_in_sample_rate,
            audio_out_sample_rate=self._params.audio_out_sample_rate,
            enable_metrics=self._params.enable_metrics,
            enable_tracing=self._enable_tracing,
            enable_usage_metrics=self._params.enable_usage_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb,
            tracing_context=self._tracing_context,
        )
        start_frame.metadata = self._create_start_metadata()
        await self._pipeline.queue_frame(start_frame)

        # Wait for the pipeline to be started before pushing any other frame.
        running = await self._wait_for_pipeline_start(start_frame)

        if running and self._params.enable_metrics and self._params.send_initial_empty_metrics:
            await self._pipeline.queue_frame(self._initial_metrics_frame())

        # A pipeline that never started can't process anything we push into it,
        # so skip straight to cleanup.
        cleanup_pipeline = True
        while running:
            frame = await self._push_queue.get()
            await self._pipeline.queue_frame(frame)
            if isinstance(frame, (CancelFrame, EndFrame, StopFrame)):
                await self._wait_for_pipeline_end(frame)
            running = not isinstance(frame, (CancelFrame, EndFrame, StopFrame))
            cleanup_pipeline = not isinstance(frame, StopFrame)
            self._push_queue.task_done()
        await self._cleanup(cleanup_pipeline)
```

이건 아홉 줄짜리 `while` loop이고 engine 전부입니다. `_push_queue`에서 pop하고, pipeline에 넘기고, frame이 terminal이었으면 반대쪽 끝으로 나오길 기다렸다가 멈춥니다.

`StartFrame`도 나머지와 같은 취급을 받습니다 — `queue_frame` 후 wait — 다만 그 wait가 bounded입니다:

**`src/pipecat/pipeline/worker.py:1039-1061`**
```python
    async def _wait_for_pipeline_start(self, frame: Frame) -> bool:
        """Wait for the specified start frame to reach the end of the pipeline.

        Returns:
            Whether the pipeline started. A pipeline that doesn't start within
            ``start_timeout_secs`` is torn down, since nothing pushed into it
            afterwards would be processed.
        """
        logger.debug(f"{self}: Starting. Waiting for {frame} to reach the end of the pipeline...")
        try:
            await asyncio.wait_for(
                self._pipeline_start_event.wait(), timeout=self._start_timeout_secs
            )
        except TimeoutError:
            logger.error(
                f"{self}: timeout waiting for {frame} to reach the end of the pipeline "
                "(being blocked somewhere?), stopping the pipeline."
            )
            await self._call_event_handler("on_pipeline_timeout", frame)
            return False
        self._pipeline_start_event.clear()
        logger.debug(f"{self}: {frame} reached the end of the pipeline, pipeline is now ready.")
        return True

```

`"being blocked somewhere?"`는 여러분이 grep하게 될 log 줄입니다. 뜻은: processor 중 하나가 `StartFrame`을 받고 20초 안에 앞으로 push하지 않았다. `StartFrame`에 대한 `process_frame`은 **input task** 위에서 돌기 때문에(§4.2 — `StartFrame`은 `SystemFrame`, `frames.py:924`), 자기 `StartFrame` 분기에서 느린 일을 하는 processor는 start 전체를 block합니다. [[frame-processor]]의 guideline이 정확히 이것입니다: system frame에 대해 `process_frame()`에서 느린 일을 절대 하지 마라.

그리고 형제 격 실패 모드가 있는데, 조용하기 때문에 더 나쁩니다: custom processor가 `process_frame`을 override하면서 `await super().process_frame(frame, direction)`을 잊으면, base 구현이 결코 실행되지 않고, `__start()`가 발화하지 않고, `__create_process_task()`가 일어나지 않고 — 그 processor는 data frame을 단 하나도 조용히 처리하지 않습니다. error도 안 납니다. 그냥 귀머거리가 됩니다. [[ch-12/read]]에서 rule-layer processor를 작성할 때, 그 한 줄이 작동과 미스터리한 무반응의 차이입니다.

### 6.3 여러분의 pipeline은 실행되는 pipeline이 아니다

`PipelineWorker`에 넘기는 list는 실행되는 것이 아닙니다. 들어가는 길에 한 번 또는 두 번 다시 감싸집니다.

**`src/pipecat/pipeline/worker.py:522-537`**
```python
        if bridged is not None:
            edge_source = _BusEdgeProcessor(
                worker=self,
                direction=FrameDirection.UPSTREAM,
                bridges=bridged,
                exclude_frames=exclude_frames,
                name=f"{self}::EdgeSource",
            )
            edge_sink = _BusEdgeProcessor(
                worker=self,
                direction=FrameDirection.DOWNSTREAM,
                bridges=bridged,
                exclude_frames=exclude_frames,
                name=f"{self}::EdgeSink",
            )
            pipeline = Pipeline([edge_source, pipeline, edge_sink])
```

**`src/pipecat/pipeline/worker.py:543-549`**
```python
        source = PipelineSource(self._source_push_frame, name=f"{self}::Source")
        self._sink = PipelineSink(self._sink_push_frame, name=f"{self}::Sink")
        # Only prepend the RTVIProcessor if we created it ourselves. When the
        # user already placed it inside their pipeline we must not insert it
        # again or it will appear twice in the frame chain.
        processors = [self._rtvi, pipeline] if prepend_rtvi else [pipeline]
        self._pipeline = Pipeline(processors, source=source, sink=self._sink)
```

그러므로 `enable_rtvi=True`(default, `worker.py:289`)일 때 여러분의 일곱 개짜리 processor list는 실제로 이렇게 돌아갑니다:

```
Pipeline([ RTVIProcessor, Pipeline([ your 7 ]) ], source=Source, sink=Sink)
```

그리고 `bridged`가 설정되어 있으면 envelope이 하나 더. 여러분의 `Pipeline`은 runtime에 두세 단계 깊이에 있습니다.

이건 framework 자신이 [[ch-01/read]]의 associativity 법칙을 현금화하는 장면입니다. worker는 여러분의 pipeline을 다른 pipeline 안에 nesting하고 frame semantics는 그대로인데, `Pipeline`이 `FrameProcessor`이고 — §4.4에 따라 — container가 queue를 추가하지 않기 때문입니다. nesting이 queue hop 하나를 요구했다면, `enable_rtvi=True`는 모든 통화에 조용히 latency를 더했을 것이고 아무도 쓰지 않았을 겁니다. 공짜라서 default로 켜져 있는 겁니다.

Lina에 대한 실무적 귀결: Pipecat log를 읽다가 `PipelineWorker#0::Source`라는 processor나 만든 적 없는 `RTVIProcessor`가 보여도 잘못된 게 없습니다. 그리고 [[ch-11/read]]이 latency budget을 위해 hop을 셀 때, 그 수는 7이 아닙니다.

---

## 7. Exit 둘 — 고객이 문장 중간에 끊는다

14:03:22. Lina가 "그럼 제가 자세한 안내를 문자로…"의 네 단어째인데 회선이 죽습니다. transport의 disconnect handler가 발화합니다. 무엇을 호출하시겠습니까?

**`src/pipecat/pipeline/worker.py:739-746`**
```python
    async def cancel(self, *, reason: str | None = None):
        """Request the running pipeline to cancel.

        Args:
            reason: Optional reason to indicate why the pipeline is being cancelled.
        """
        if not self._finished:
            await self._cancel(reason=reason)
```

signature를 보세요: `async def cancel(self, *, reason: str | None = None)`. `*` 때문에 `reason`은 **keyword-only**입니다. `await worker.cancel("hangup")`은 cancel-with-reason이 아니라 `TypeError`입니다. `runner.py:350`의 `WorkerRunner.cancel(self, reason=None)`은 *positional-or-keyword*이므로 둘은 call-compatible하지 않고, 둘 사이를 refactoring할 때 저지르기 쉬운 실수입니다. 항상 `reason=`으로 쓰세요.

**`src/pipecat/pipeline/worker.py:973-984`**
```python
    async def _cancel(self, *, reason: str | None = None):
        """Internal cancellation logic for the pipeline worker.

        Args:
            reason: Optional reason to indicate why the pipeline is being cancelled.
        """
        if not self._cancelled:
            logger.debug(f"Cancelling pipeline worker {self}")
            self._cancelled = True
            if not self._pipeline_start_event.is_set():
                self._pipeline_start_event.set()
            await self.queue_frame(CancelFrame(reason=reason))
```

세 동작. `_cancelled` 설정. **start event 해제** — 이게 아직 `_wait_for_pipeline_start`에 갇힌 통화, 즉 provider handshake가 느린 동안 끊어버린 고객을 cancel할 수 있게 해줍니다. 그다음 `CancelFrame` queue.

`CancelFrame`은 `SystemFrame`이므로(`frames.py:999`) 모든 processor에서 §4.2의 out-of-band 경로를 탑니다: priority 10, input task 위에서 inline 실행, queue에 있는 75개 audio chunk보다 앞서서. 각 processor에서 base `process_frame` dispatch에 도달해 `__cancel`을 호출합니다:

**`src/pipecat/processors/frame_processor.py:1099-1106`**
```python
    async def __cancel(self, frame: CancelFrame):
        """Handle the cancel frame to stop processor operation.

        Args:
            frame: The cancel frame.
        """
        self._cancelling = True
        await self.__cancel_process_task()
```

`self._cancelling = True` — 이제 §4.1을 떠올리세요: 이 순간부터 `queue_frame`은 조기 반환하고 그 processor의 이후 모든 frame은 **조용히 drop**됩니다. 그다음 process task가 `await` 도중에 죽습니다. queue에 있던 한국어 audio는 재생되지 않습니다. 그게 요점입니다: 고객은 갔고, audio는 갈 곳이 없고, 그걸 drain하려고 session을 열어두는 건 순수한 비용입니다.

대기는 bounded입니다:

**`src/pipecat/pipeline/worker.py:1063-1095`**
```python
    async def _wait_for_pipeline_end(self, frame: Frame):
        """Wait for the specified frame to reach the end of the pipeline."""

        async def wait_for_cancel():
            try:
                await asyncio.wait_for(
                    self._pipeline_end_event.wait(), timeout=self._cancel_timeout_secs
                )
                logger.debug(f"{self}: {frame} reached the end of the pipeline.")
            except TimeoutError:
                logger.warning(
                    f"{self}: timeout waiting for {frame} to reach the end of the pipeline (being blocked somewhere?)."
                )
                await self._call_event_handler("on_pipeline_timeout", frame)
            finally:
                await self._call_event_handler("on_pipeline_finished", frame)

        logger.debug(f"{self}: Closing. Waiting for {frame} to reach the end of the pipeline...")

        if isinstance(frame, CancelFrame):
            await wait_for_cancel()
        else:
            # Ending flushes what is queued, so cutting the wait short would
            # drop the audio the EndFrame exists to play out. A processor that
            # could hold it up watches for that itself.
            await self._pipeline_end_event.wait()
            logger.debug(f"{self}: {frame} reached the end of the pipeline, pipeline is closing.")

        self._pipeline_end_event.clear()

        # We are really done. Setting ``_finished_event`` makes
        # ``BaseWorker.wait()`` resolve for callers awaiting this worker.
        self._finished_event.set()
```

`CancelFrame` → `wait_for_cancel()` → `asyncio.wait_for(..., timeout=self._cancel_timeout_secs)`, default는 `CANCEL_TIMEOUT_SECS = 20.0`. 어떤 processor의 `CancelFrame` handler가 박히면 — `CancelledError`를 삼켜버리는 vendor SDK, 죽은 socket에서 block되는 `close()` — worker는 20초 후 포기하고, `on_pipeline_timeout`을 발화시키고, `finally`에서 **그럼에도** `on_pipeline_finished`를 발화시킵니다. session은 어느 쪽이든 끝납니다. 그 degradation이 bound가 존재하는 이유입니다: 박힌 teardown이, 하루 종일 새 통화를 받는 host에서 worker 하나를 영원히 붙들어두면 안 됩니다.

Lina 기준, 고객이 이미 떠난 뒤 20초 동안 resource를 붙잡는 건 후하지만 견딜 만합니다. 동시 통화 200건 지점에 가면 `cancel_timeout_secs`(`worker.py:283`)가 knob이고, 5초 bound에 `on_pipeline_timeout` alert를 거는 게 방어 가능한 자세입니다.

---

## 8. Exit 셋 — bot이 마지막 멘트를 끝낸다

다른 상황, 같은 통화. 고객이 "네, 그럼 문자로 보내주세요"라고 했고, Lina가 마지막 멘트를 하고 있고, 그게 끝나면 *여러분*이 통화를 깔끔히 종료하고 싶습니다. 고객은 아직 회선에 있습니다. audio가 중요합니다.

**`src/pipecat/pipeline/worker.py:730-737`**
```python
    async def stop_when_done(self):
        """Schedule the pipeline to stop after processing all queued frames.

        Sends an EndFrame to gracefully terminate the pipeline once all
        current processing is complete.
        """
        logger.debug(f"Task {self} scheduled to stop when done")
        await self.queue_frame(EndFrame())
```

동작 한 줄: `EndFrame`을 queue한다. 그리고 `EndFrame`은 의도적으로 `SystemFrame`이 **아닙니다**:

**`src/pipecat/frames/frames.py:1899`**
```python
class EndFrame(ControlFrame, UninterruptibleFrame):
```

`ControlFrame`(`frames.py:128`)이지 `SystemFrame`(`frames.py:105`)이 아닙니다. `FrameProcessorQueue.put`에서 `DEFAULT_PRIORITY = 20`을 받습니다. audio와 함께 `__process_queue`로 들어갑니다. 모든 processor에서 Lina 마지막 멘트의 모든 chunk **뒤에** 줄 서고, 마지막 sample이 write된 뒤에야 output transport에 도착합니다.

그게 설계입니다. `EndFrame` docstring이 source에서 그렇게 말합니다: *"Note, that this is a control frame, which means it will be received in the order it was sent."* in-band인 이유는 **in-band인 것이 feature이기 때문입니다.** audio를 추월했다면 Lina가 말하는 중간에 transport를 tear down했을 것이고 고객은 딸깍 소리를 들었을 겁니다.

그리고 §4.3의 `UninterruptibleFrame` mixin이 그걸 견딜 수 있게 만듭니다: `EndFrame`이 아직 queue에 있는 동안 barge-in이 발화하면, `FrameQueue.reset()`은 audio를 flush하고 `EndFrame`은 **유지**합니다. docstring이 이유를 말합니다: *"Terminal frames must survive interruption to guarantee proper pipeline shutdown."* mixin이 없었다면 teardown 중의 interrupt가 여러분의 shutdown signal을 지웠을 것이고, worker는 다른 무언가가 죽일 때까지 매달렸을 겁니다.

그다음 대기. §7의 `_wait_for_pipeline_end`로 돌아가 `else` 분기를 읽으세요:

```python
        else:
            # Ending flushes what is queued, so cutting the wait short would
            # drop the audio the EndFrame exists to play out. A processor that
            # could hold it up watches for that itself.
            await self._pipeline_end_event.wait()
```

맨 `await`. `wait_for` 없음. timeout 없음. **설계상 unbounded, 영원히**, 그리고 comment가 근거를 밝힙니다: 여기 timeout을 걸면 `EndFrame`이 재생하려고 존재하는 바로 그 audio를 drop하게 되고, 그건 graceful path의 목적 전체를 무너뜨립니다.

그래서 비대칭을, 외울 것으로 정리하면:

| | frame class | priority tier | 들어가는 queue | 대기 |
|---|---|---|---|---|
| `cancel()` | `CancelFrame(SystemFrame)` | 10 | 없음 — input task 위 inline | **bounded**, `cancel_timeout_secs` = 20 s |
| `stop_when_done()` | `EndFrame(ControlFrame, UninterruptibleFrame)` | 20 | `__process_queue`, audio 뒤 | **unbounded** |

violent path는 bounded; graceful path는 아닙니다. 이유는 대칭성이나 취향이 아니라 — violent path에는 지킬 게 남아 있지 않고 graceful path는 *오직* 무언가를 지키려고 존재하기 때문입니다. "당연히 graceful shutdown 쪽에 timeout이 있겠지"라는 모든 직관은 여기서 뒤집힙니다.

Lina host에 대한 운영상 귀결은 직접적이고, 이를 전제로 설계해야 합니다: **`stop_when_done()`은 영원히 매달릴 수 있습니다.** processor 하나가 `EndFrame`을 앞으로 push하지 않으면 — 영원히 답하지 않을 vendor socket을 기다리며 박힌 TTS service — 그 worker는 결코 끝나지 않고, entry는 `_entries`(§3.1)를 떠나지 않고, framework의 어떤 것도 timeout을 걸지 않습니다. host 차원의 방어는 watchdog입니다: graceful path를 `asyncio.wait_for(worker.wait(), timeout=...)`로 감싸고, `worker.cancel(reason="drain timeout")`으로 fallback. Pipecat은 그걸 제공하지 않습니다. §13이 그걸 topology에 넣습니다.

pipeline을 끝내지 않고 실제로 drain되었는지 알고 싶다면 probe도 있습니다:

**`src/pipecat/pipeline/worker.py:831-855`**
```python
    async def flush_pipeline(self, timeout: float = 5.0) -> bool:
        """Flush all in-flight frames from the pipeline and wait for it to drain.

        Pushes a :class:`~pipecat.frames.frames.PipelineFlushFrame` downstream;
        the sink bounces it back upstream and the source sets its event once it
        completes the round-trip, signalling that every frame queued ahead of it
        has been processed. The probe is injected straight into the pipeline so
        it bypasses any ``queue_frame`` override (e.g. tool-call deferral).

        Args:
            timeout: Seconds to wait before giving up. On timeout a warning is
                logged and ``False`` is returned rather than blocking forever
                (e.g. if a processor swallows the probe).

        Returns:
            True if the pipeline drained, False if the wait timed out.
        """
        event = asyncio.Event()
        await self._pipeline.queue_frame(PipelineFlushFrame(event=event))
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except TimeoutError:
            logger.warning(f"{self}: pipeline flush timed out after {timeout}s")
            return False
```

한 파일 안에서 engineering 자세가 대비되는 점에 주목하세요: 이 method는 timeout이 **있고** 매달리는 대신 `False`를 반환하며, docstring이 이유를 말합니다 — "rather than blocking forever (e.g. if a processor swallows the probe)." framework는 wait를 bound하는 법을 압니다. `EndFrame` 쪽은 bound하지 않기로 *선택한* 겁니다.

round trip이 작동하는 건 두 번째 out-of-band axis인 direction 덕분입니다. sink가 probe를 **upstream**으로 튕기고, worker의 source가 그걸 잡습니다:

**`src/pipecat/pipeline/worker.py:1259-1266`**
```python
        if isinstance(frame, PipelineFlushFrame):
            # The flush probe completed its round-trip (down to the sink, back up
            # to the source). Everything queued ahead of it has been processed;
            # release whoever is awaiting it.
            logger.debug(f"{self}: flush probe reached source — pipeline drained")
            if frame.event:
                frame.event.set()
            return
```

그 upstream 반환 경로는 pipeline이 worker에게 스스로 shut down하라고 요청하는 방법이기도 합니다. `_source_push_frame`(`worker.py:1248-1297`)은 upstream worker frame을 downstream lifecycle frame으로 번역합니다 — `EndWorkerFrame` → `EndFrame`, `CancelWorkerFrame` → `CancelFrame`, `StopWorkerFrame` → `StopFrame`. 그리고 그중 하나는 push queue를 통째로 우회하는데, 그 comment는 그 자체로 작은 교훈입니다:

**`src/pipecat/pipeline/worker.py:1280-1286`**
```python
        elif isinstance(frame, InterruptionWorkerFrame):
            # Tell the worker we should interrupt the pipeline. Note that we are
            # bypassing the push queue and directly queue into the
            # pipeline. This is in case the push worker is blocked waiting for a
            # pipeline-ending frame to finish traversing the pipeline.
            logger.debug(f"{self}: received interruption worker frame upstream {frame}")
            await self._pipeline.queue_frame(InterruptionFrame())
```

이유를 읽으세요: push task가 그 unbounded `EndFrame` 대기에 앉아 있을 수 있습니다. interruption이 `_push_queue`를 거쳐야 했다면 drain 중에는 결코 도착할 수 없습니다. 그래서 pipeline으로 곧장 들어갑니다. 그게 *out-of-band 원리를 한 단계 위에 적용한 것*입니다 — §5.1과 같은 논증, 다른 queue.

---

## 9. Exit 넷 — 고객이 침묵한다

고객이 "여보세요"라고 했고, Lina의 오프닝을 들었고, 그다음… 아무것도. 끊지 않았습니다. 회선은 열려 있습니다. 4분이 지납니다.

아무 일도 일어나지 않습니다. 이유는 이렇고, 여러분에게 틀린 숫자는 이것입니다.

배관은 세 부분입니다. 첫째, 활동을 알아채는 observer:

**`src/pipecat/pipeline/worker.py:106-140`**
```python
class IdleFrameObserver(BaseObserver):
    """Idle timeout observer.

    This observer waits for specific frames being generated in the pipeline. If
    the frames are generated the given asyncio event is set. If the event is not
    set it means the pipeline is probably idle.

    """

    def __init__(self, *, idle_event: asyncio.Event, idle_timeout_frames: tuple[type[Frame], ...]):
        """Initialize the observer.

        Args:
            idle_event: The event to set if the idle timeout frames are being pushed.
            idle_timeout_frames: A tuple with the frames that should set the event when received
        """
        super().__init__()
        self._idle_event = idle_event
        self._idle_timeout_frames = idle_timeout_frames
        self._processed_frames = set()

    async def on_push_frame(self, data: FramePushed):
        """Callback executed when a frame is pushed in the pipeline.

        Args:
            data: The frame push event data.
        """
        # Skip already processed frames
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)

        if isinstance(data.frame, StartFrame) or isinstance(data.frame, self._idle_timeout_frames):
            self._idle_event.set()
```

자동으로 설치되지만, timeout이 설정된 경우에만입니다:

**`src/pipecat/pipeline/worker.py:500-506`**
```python
        self._idle_event = asyncio.Event()
        self._idle_monitor_task: asyncio.Task | None = None
        if self._idle_timeout_secs:
            idle_frame_observer = IdleFrameObserver(
                idle_event=self._idle_event,
                idle_timeout_frames=idle_timeout_frames,
            )
            observers.append(idle_frame_observer)
```

둘째, 그 event를 timeout과 함께 기다리는 monitor task:

**`src/pipecat/pipeline/worker.py:1401-1415`**
```python
    async def _idle_monitor_handler(self):
        """Monitor pipeline activity and detect idle conditions.

        Tracks frame activity and triggers idle timeout events when the
        pipeline hasn't received relevant frames within the timeout period.

        Note: Heartbeats are excluded from idle detection.
        """
        running = True
        while running:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._idle_timeout_secs)
                self._idle_event.clear()
            except TimeoutError:
                running = await self._idle_timeout_detected()
```

셋째, 조치:

**`src/pipecat/pipeline/worker.py:1417-1441`**
```python
    async def _idle_timeout_detected(self) -> bool:
        """Handle idle timeout detection and optional cancellation.

        Returns:
            Whether the pipeline worker should continue running.
        """
        # If we are cancelling, just exit the worker.
        if self._cancelled:
            return False

        logger.warning("Idle timeout detected.")
        await self._call_event_handler("on_idle_timeout")
        if not self._cancel_on_idle_timeout:
            return True

        logger.warning("Idle pipeline detected, cancelling pipeline worker...")
        await self.cancel(reason="idle timeout")
        if self._cancel_runner_on_idle_timeout:
            logger.warning("...and cancelling the runner.")
            # ``BaseWorker.cancel`` sends ``BusCancelMessage`` on the bus
            # so the runner broadcasts cancellation to every other root
            # worker too. This worker's pipeline is already cancelling
            # from the call above.
            await BaseWorker.cancel(self, reason="idle timeout")
        return False
```

### 9.1 default들, 그리고 Lina 기준으로 잘못된 두 가지 전부

**`src/pipecat/pipeline/worker.py:281-292`** (keyword-only constructor signature `:273-303`의 발췌)
```python
        cancel_on_idle_timeout: bool = True,
        cancel_runner_on_idle_timeout: bool = True,
        cancel_timeout_secs: float = CANCEL_TIMEOUT_SECS,
        check_dangling_tasks: bool = True,
        clock: BaseClock | None = None,
        conversation_id: str | None = None,
        enable_tracing: bool = False,
        enable_turn_tracking: bool = True,
        enable_rtvi: bool = True,
        exclude_frames: tuple[type[Frame], ...] | None = None,
        idle_timeout_frames: tuple[type[Frame], ...] = (BotSpeakingFrame, UserSpeakingFrame),
        idle_timeout_secs: float | None = IDLE_TIMEOUT_SECS,
```

**문제 하나: 300초.** 한국 outbound 보험 dial에서 5분의 dead air. 30초면 이미 놓친 고객이고; 60초면 수화기를 테이블에 놓고 자리를 뜬 겁니다. 5분은 5분치 telephony 요금, STT websocket, TTS session, 그리고 이미 떠난 고객을 위해 warm하게 유지하느라 돈을 내는 LLM context입니다. default는 여러분의 product에 맞춰 calibrate된 게 아니고, 그걸 옳다고 볼 수 있는 해석은 없습니다.

**문제 둘, 이건 진짜 위험입니다: `cancel_runner_on_idle_timeout=True`.** `_idle_timeout_detected`를 다시 읽으세요. idle 시 worker를 cancel하고 — 여기까진 좋습니다 — 그다음 `BaseWorker.cancel(self, ...)`을 호출하는데, comment가 그것이 "sends `BusCancelMessage` on the bus so the runner broadcasts cancellation to every other root worker too"라고 알려줍니다.

`auto_end=False`이고 동시 통화 30건인 Lina host에서, **침묵하는 고객 한 명이 나머지 스물아홉 명을 내려버립니다.** 이건 §3.2의 `runner.cancel()` 위험과 같은 모양이고 더 음험한데, 아무도 아무것도 호출하지 않았는데 발화하기 때문입니다: 5분 동안 아무 말 안 한 고객이 host 전체 cancel을 trigger합니다. 이 섹션에서 한 줄만 가져간다면 이걸 가져가세요: multi-session host에서 `cancel_runner_on_idle_timeout=False`는 필수입니다.

재튜닝은 constructor kwarg이지, patch도 subclass도 아닙니다:

```python
worker = PipelineWorker(
    pipeline,
    idle_timeout_secs=45.0,                       # sales-call dead air, not a browser demo
    cancel_on_idle_timeout=True,                  # end this call
    cancel_runner_on_idle_timeout=False,          # NEVER on a multi-session host
    idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame),
)
```

### 9.2 무엇이 clock을 reset하는가 — 틀리기 쉬운 부분

`idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame)`은 "turn이 일어나면 reset"을 뜻하지 않습니다. 두 frame 모두 **speech가 진행되는 동안 주기적으로**, 200 ms cadence로 방출됩니다.

**`src/pipecat/transports/base_output.py:459-463`**
```python
            # Last time a BotSpeakingFrame was pushed.
            self._bot_speaking_frame_time = 0
            # How often a BotSpeakingFrame should be pushed (value should be
            # greater than the audio chunks to have any effect).
            self._bot_speaking_frame_period = 0.2
```

**`src/pipecat/transports/base_output.py:774-781`**
```python
        async def _bot_currently_speaking(self):
            """Handle bot speaking event."""
            await self._bot_started_speaking()

            diff_time = time.time() - self._bot_speaking_frame_time
            if diff_time >= self._bot_speaking_frame_period:
                await self._transport.broadcast_frame(BotSpeakingFrame)
                self._bot_speaking_frame_time = time.time()
```

그리고 user 쪽은, VAD processor 자신의 docstring에서 — *"`UserSpeakingFrame`: Pushed periodically while speech is detected"* (`processors/audio/vad_processor.py:34`) — `speech_activity_period: float = 0.2` (`:45`)와 `:86`의 push로.

`BotSpeakingFrame`의 docstring이 그것이 default list에 있는 정확한 이유를 짚습니다:

**`src/pipecat/frames/frames.py:1304-1311`**
```python
class BotSpeakingFrame(SystemFrame):
    """Frame indicating the bot is currently speaking.

    Emitted upstream and downstream by the BaseOutputTransport while the bot is
    still speaking. This can be used, for example, to detect when a user is
    idle. That is, while the bot is speaking we don't want to trigger any user
    idle timeout since the user might be listening.
    """
```

그러므로 clock은 **진짜 상호 침묵**을 잽니다: 양쪽 모두 audio를 만들지 않는 상태가 `idle_timeout_secs` 동안 연속으로. 90초짜리 Lina 독백을 듣고 있는 고객은 idle이 아닙니다, `BotSpeakingFrame`이 그동안 내내 초당 다섯 번 발화하니까요. 그게 옳은 semantics이고, 45초가 진짜로 아무도 말하지 않는 45초 — 바쁜 통화 중의 긴 pause가 아니라 진짜 dead air — 임을 뜻합니다.

> 💡 **쉬운 설명 — 왜 "주기적"이어야 하나**
> `BotStartedSpeakingFrame`(발화 시작 시 1회)만 list에 있었다면, 90초 독백은 t=0에 clock을 한 번 reset하고
> t=45에 idle timeout을 때렸을 겁니다 — bot이 말하는 도중에 통화가 끊깁니다. 200 ms cadence로 계속 나오는
> `BotSpeakingFrame`은 clock을 "발화가 시작됐다"가 아니라 "발화가 아직 진행 중이다"의 heartbeat으로 만듭니다.
> idle timer는 event 기반이 아니라 liveness 기반이어야 한다는 뜻입니다.

이건 tuple을 어떻게 *좁힐지*도 알려줍니다. timer가 "고객이 말하지 않았다"를 재기를 원한다면, `BotSpeakingFrame`을 빼고 `idle_timeout_frames=(UserSpeakingFrame,)`을 넘기세요. 그러면 Lina의 독백은 clock을 reset하지 않고, 응답을 멈춘 고객은 bot이 아직 말하는 중에도 감지됩니다. 그건 다른 product decision이고 — 방치된 수화기를 감지하는 데는 아마 그게 옳은 쪽이고 — code 변경이 아니라 tuple 한 줄 변경입니다.

알아둘 만한 경계 조건 하나: `IdleFrameObserver.on_push_frame`은 `StartFrame`에서도 event를 set하므로, clock은 construction이 아니라 pipeline start에서 시작합니다; 그리고 `_maybe_start_idle_task()`는 `_process_push_queue`의 맨 위(`worker.py:1212`)에서 호출되므로 setup 동안에는 idle monitoring이 존재하지 않습니다. connect 중에 매달리는 provider는 idle timer가 아니라 `SETUP_TIMEOUT_SECS`의 문제(§6.1)입니다.

여러분이 이미 가진 것과 어떻게 매핑되는지 기록: [[boson-gateway-server]]에 따르면 boson의 endpointer는 `_start_silence_timer`(`websocket.py:616`)이고, `silence_timeout_ms / 1000`(default `2000`)만큼 sleep한 뒤 `_finalize_partial`(`:661`)을 호출합니다. 그 timer는 2초에서 동작하는 **turn-boundary** 장치입니다 — user의 발화가 끝났는지를 결정합니다. Pipecat의 300초 idle timeout은 **session-abandonment** 장치입니다. 둘은 서로 다른 질문에 답하는 서로 다른 clock이고, 하나를 다른 하나에 직접 매핑하는 건 category error입니다; boson silence timer의 대응물은 `src/pipecat/turns/`에 있고 그건 [[ch-06/read]]의 영역입니다.

---

## 10. `queue_frame`의 direction 분기, 그리고 이후 chapter들이 그걸 필요로 하는 이유

mechanism 하나 더, 작지만 이후 두 chapter의 경첩입니다.

**`src/pipecat/pipeline/worker.py:793-808`**
```python
    async def queue_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Queue a single frame to be pushed through the pipeline.

        Downstream frames are pushed from the beginning of the pipeline.
        Upstream frames are pushed from the end of the pipeline.

        Args:
            frame: The frame to be processed.
            direction: The direction to push the frame. Defaults to downstream.
        """
        if direction == FrameDirection.DOWNSTREAM:
            await self._push_queue.put(frame)
        else:
            await self._sink.queue_frame(frame, direction)
```

direction에 따라 서로 다른 injection point 두 개:

- **DOWNSTREAM** → `self._push_queue`, `_process_push_queue`가 drain해서 `self._pipeline.queue_frame(...)`으로 넣습니다. frame은 **head**로 들어가 모든 processor를 통과합니다.
- **그 외** → `self._sink.queue_frame(frame, direction)`. frame은 **tail**로 들어가 거꾸로 갑니다.

두 번째 분기가 `elif direction == UPSTREAM`이 아니라 `else`이고, `FrameDirection`은 member가 정확히 둘(`frame_processor.py:60-69`)이므로 동작은 total합니다.

이게 course 뒤쪽에서 중요한 이유:

- [[ch-10/read]]: `FlowManager`는 pipeline **밖에** 살면서 frame을 주입해 pipeline을 구동합니다. frame이 어느 끝으로 들어가느냐가 어떤 processor가 그것을 보는지를 결정하고, 따라서 node transition이 LLM보다 먼저 aggregator에 관측되는지 나중인지를 결정합니다.
- [[ch-12/read]]: transition race. 서로 다른 방향에서 in-flight인 두 injection에는 global order가 없습니다 — §4.2는 processor당 ordering만 줬습니다. 이 system에 global ordering은 없고, direction 분기는 그게 추상이기를 그만두는 지점입니다.
- [[ch-09/read]]: assistant aggregator는 `transport.output()` 뒤에 앉아 있고 context를 LLM 쪽으로 **거꾸로** push해야 합니다. upstream traffic은 이국적인 케이스가 아니라 대화 loop가 닫히는 방식입니다.

worker의 편의 wrapper에 대해서도: `queue_frames`(`worker.py:810-829`)는 `Iterable`이나 `AsyncIterable`을 받아서 단순히 loop를 돌며 item마다 `queue_frame`을 호출합니다. batching도 atomicity도 추가하지 않습니다 — `queue_frames` 호출에서 나온 frame들은 다른 곳에서 온 frame들과 interleave될 수 있습니다.

---

## 11. canonical pipeline, data-dependency chain으로 읽기

runtime은 다 봤습니다. 이제 모든 게 돌아가는 모양, 그리고 그것을 call sequence가 아니라 **dependency graph**로 읽어야 하는 이유입니다.

**`examples/getting-started/06-voice-agent.py:81-91`**
```python
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )
```

[[canonical-voice-bot]]에 따르면 이 정확한 일곱 항목 list는 `06a-voice-agent-local.py`(L69–73), `07-function-calling.py`(L110–114), 그리고 `examples/voice/voice-*.py` 전반에 그대로 등장합니다 — 한 example의 선택이 아니라 house pattern입니다.

모든 위치는 **그 지점에서만 존재하는 증거**에 의해 고정됩니다:

1. `transport.input()` — inbound audio의 유일한 producer. upstream에 아무것도 없습니다.
2. `stt` — raw audio가 필요하고, text를 생산합니다. audio와 text 위에서 추론하는 무엇 사이에 있어야 합니다.
3. `user_aggregator` — transcription이 필요하고; turn이 끝나면 user message를 공유 `LLMContext`에 쓰고 context frame을 downstream으로 push합니다. **그 downstream push가 LLM trigger**이므로, 이것은 LLM 바로 앞에 있어야 합니다.
4. `llm` — context frame을 소비하고, streaming text를 방출합니다.
5. `tts` — text가 필요하고, audio를 생산합니다. LLM 뒤, output transport 앞.
6. `transport.output()` — *실제로 재생된 것*을 아는 유일한 component.
7. `assistant_aggregator` — 실제 playback 경계에서의 `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame`이 필요합니다. `transport.output()` 앞에 두면 고객이 듣지 않은 text를 commit합니다.

마지막 것이 사람들이 놀라는 위치이자 sales script에 가장 중요한 위치입니다. barge-in은 assistant turn을 *생성된* 지점이 아니라 *도달한* 지점에서 truncate합니다 — 그게 CRM에 "65세 갱신 옵션을 제안했음"이 기록되는 것과 고객이 실제로 그걸 들은 것의 차이입니다.

그리고 entry point — 처음 접할 때 정말로 반직관적인 것:

**`examples/getting-started/06-voice-agent.py:107-114`**
```python
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])
```

**frame이 queue되기 전까지 아무것도 실행되지 않습니다.** pipeline이 생성되었고, worker가 추가되었고, `runner.run()`이 await되었고, `StartFrame`이 모든 processor를 통과했고 모든 processor가 start되어 듣고 있는데 — bot은 아무 말도 하지 않습니다. 먼저 말해야 하는 voice agent, 즉 모든 outbound sales call은, 오직 transport event handler에서 `queue_frames([LLMRunFrame()])`이 발화했기 때문에 첫 단어를 냅니다. pipeline은 기계이고; 대화는 여러분이 시작하는 무엇입니다.

Lina 기준으로 거기가 여러분의 오프닝 script가 들어가는 자리이고, [[boson-gateway-server]]에 따르면 session-identity 질문이 착지하는 자리이기도 합니다 — Pipecat pipeline은 per-connection이므로, boson의 reconnect-and-resume 동작(disconnect를 넘어 살아남는 session, 1800 s idle TTL, `SessionAccess.authorize`)은 route를 소유하는 바깥쪽 FastAPI layer에 살아야 합니다. §13이 그 layer를 topology에 명시적으로 넣습니다.

**`ParallelPipeline`은 존재하고**(`src/pipecat/pipeline/parallel_pipeline.py:24`, `class ParallelPipeline(BasePipeline)`) 이 course는 그것을 쓰지 않습니다.

---

## 12. framework가 주지 *않는* 것

짧은 섹션입니다, [[deployment-scaling]]이 디테일을 나르고 [[ch-13/read]]이 회계를 하기 때문입니다. 하지만 §13의 deliverable은 경계를 알아야만 정직합니다.

번들된 runner는 자기 source에서 development라고 라벨링합니다: `runner/run.py`의 banner는 `ᓚᘏᗢ PIPECAT DEVELOPMENT RUNNER`를 출력합니다. 모든 session을 하나의 loop 위 `asyncio.Task`로 multiplexing합니다:

**`src/pipecat/runner/run.py:215-220`**
```python
def _start_bot_session(coro) -> asyncio.Task:
    """Run a bot in the background, holding a reference until it finishes."""
    task = asyncio.create_task(coro)
    _bot_sessions.add(task)
    task.add_done_callback(_bot_sessions.discard)
    return task
```

`:212`의 module-level set이 존재하는 이유는 `:209-211`의 comment가 말합니다: *"the event loop only holds a weak reference to a task, so one that nothing else references can be collected while it is still running."* 그건 FastAPI route에서 per-session task를 spawn할 때 여러분이 직접 부딪히게 될 진짜 bug class입니다 — strong reference를 잡으세요.

그리고 process model은 그냥 하나의 process입니다: `run.py:1999`의 `uvicorn.run(app, host=args.host, port=args.port)`, `workers=` 인자 없음. 그러므로 concurrency는 하나의 event loop 위의 concurrent asyncio task입니다. process pool도, admission control도, per-session CPU isolation도, `runner/` 어디에도 session-count 제한도 없습니다.

거기에 §3.1의 없는 `remove_workers`와 §8의 unbounded drain을 더하면 남은 작업의 모양이 분명합니다: **Pipecat은 session runtime을 주지, host를 주지 않습니다.** host는 여러분 몫입니다.

---

## 13. Deliverable: Lina host를 위한 process / session / worker topology

이게 이 chapter의 목적이었습니다. 서베이가 아니라 결정이고, 이후 chapter들은 이걸 다시 열지 않고 그 위에 짓습니다.

### 13.1 topology

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ONE OS PROCESS  ·  uvicorn, ONE event loop, no workers= arg              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ FastAPI app (yours — owns routes, auth, session identity)          │  │
│  │   · GET /ws/{session_id}   — accepts, authorises, then hands the   │  │
│  │     accepted WebSocket to a Pipecat transport                      │  │
│  │   · session store + reconnect/resume + idle TTL   (kept, not       │  │
│  │     replaced — Pipecat has no counterpart; see §11)                │  │
│  │   · process-scoped resources: MCP subprocesses, DB pool, HTTP      │  │
│  │     clients — created ONCE at startup, above the workers           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ONE WorkerRunner, created at startup, run with auto_end=False      │  │
│  │   WorkerRunner(handle_sigint=True, handle_sigterm=True)            │  │
│  │   asyncio.create_task(runner.run(auto_end=False))   ← held ref     │  │
│  │                                                                    │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │  │
│  │   │PipelineWorker│  │PipelineWorker│  │PipelineWorker│  … per call │  │
│  │   │  call A      │  │  call B      │  │  call C      │             │  │
│  │   │  Pipeline(7) │  │  Pipeline(7) │  │  Pipeline(7) │             │  │
│  │   │  ~2 asyncio  │  │              │  │              │             │  │
│  │   │  tasks per   │  │              │  │              │             │  │
│  │   │  processor   │  │              │  │              │             │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

규칙으로 서술하면:

1. **host당 OS process 하나.** 통화당 하나가 아닙니다. `workers=` 없는 `uvicorn.run(app, ...)`. scale out은 한 container 안의 loop를 늘리는 게 아니라 container를 더 돌려서 합니다.
2. **application startup에 한 번 생성되는 `WorkerRunner` 하나**, `auto_end=False`로. §3에 따라 협상 불가입니다 — default는 첫 고객이 끊을 때 host를 종료시킵니다.
3. **통화당 `PipelineWorker` 하나**, websocket route에서 생성하고 `await runner.add_workers(worker)`로 등록. default(자동 생성) 이름이나 per-connection session UUID; 절대 전화번호는 안 됩니다(§3.1a).
4. **모든 session은 같은 loop 위의 `asyncio.Task`.** `add_workers`가 `_start_worker`를 통해 대신 해줍니다. task를 여러분이 만들지 않고; 그것을 `asyncio.cancel` 해서도 안 됩니다(§6).
5. **per-call shutdown은 `worker.*`, 절대 `runner.*`가 아닙니다.** hangup에는 `worker.cancel(reason=...)`; 깔끔한 종료에는 `worker.stop_when_done()`. `runner.cancel()`과 `runner.end()`는 host 전체(§3.2)이고 process shutdown handler에만 속합니다.
6. **process-scoped resource는 runner 위에 삽니다.** MCP subprocess, DB pool, HTTP client: `PipelineWorker.__init__`이 아니라 FastAPI startup에서, 그리고 `app_resources`(`worker.py:279`)로 내려보냅니다 — constructor docstring은 framework가 그것을 "passes through untouched" 하고 `worker.app_resources`로 노출한다고 말합니다. [[deployment-scaling]]에 따르면 boson의 `GatewayCore` process-scoped MCP subprocess 소유권은 **keep**이고, 그것을 worker 위로 re-host 하는 게 요점 전부입니다 — 그러지 않으면 통화마다 re-spawn됩니다.

### 13.2 constant들, 결정됨

| Knob | framework default | Lina | 이유 |
|---|---|---|---|
| `WorkerRunner(auto_end=...)` | `True` | **`False`** | §3 — 아니면 첫 통화가 끝날 때 host가 종료됨 |
| `WorkerRunner(handle_sigterm=...)` | `False` | **`True`** | §3.2 — container eviction은 SIGTERM을 보냄 |
| `idle_timeout_secs` | `300` | **`45`** | §9 — sales dial에서 5분 dead air는 5분치 태운 요금 |
| `cancel_on_idle_timeout` | `True` | `True` | 방치된 통화를 종료 |
| `cancel_runner_on_idle_timeout` | `True` | **`False`** | §9.1 — 아니면 침묵하는 고객 한 명이 모든 동시 통화를 죽임 |
| `idle_timeout_frames` | `(BotSpeakingFrame, UserSpeakingFrame)` | 유지, 또는 `(UserSpeakingFrame,)`로 좁힘 | §9.2 — 좁히면 "상호 침묵" 대신 "고객이 응답을 멈춤"을 잼 |
| `setup_timeout_secs` | `20.0` | **`5.0`** | §6.1 — "여보세요" 뒤 20초 침묵은 잃은 통화 |
| `start_timeout_secs` | `20.0` | `20.0` | 그대로; 여기서 걸리는 건 tunable이 아니라 여러분 processor의 bug |
| `cancel_timeout_secs` | `20.0` | 처음엔 `20.0` | §7 — concurrency 상황에서 재검토, `on_pipeline_timeout`에 alert |
| `enable_rtvi` | `True` | `True` | §6.3 — queue hop 비용이 없고, [[ch-11/read]]이 observability를 원함 |
| `processor_unusable_policy` | `CONTINUE` | [[ch-13/read]]에서 결정 | [[ch-05/read]]과 [[ch-07/read]]의 provider-failover 증거가 필요 |

### 13.3 직접 만들어야 하는 세 가지

발견되지 않고 예산에 잡히도록 지금 이름을 붙입니다.

1. **drain watchdog.** `stop_when_done()`은 unbounded입니다(§8). 감싸세요: `asyncio.wait_for(worker.wait(), timeout=N)` 그리고 `worker.cancel(reason="drain timeout")`으로 fallback. framework는 이걸 해주지 않고 kwarg도 없습니다.
2. **worker-entry 위생.** `remove_workers`가 없습니다(§3.1c). `WorkerRunner._entries`는 process 수명 동안 통화당 entry 하나씩 자라고, 각각 끝난 `PipelineWorker`와 그 `Pipeline` 전체에 대한 live reference를 쥡니다. 하나를 고르세요: 통화 수나 wall-clock 스케줄로 process를 recycle; 또는 runner 여러 개를 pool하고 은퇴시킴; 또는 실제 통화량으로 하루 전체를 측정한 뒤 그 증가를 수용. `_entries`에 손대지 마세요 — private이고 옮겨갑니다.
3. **바깥쪽 session layer.** [[boson-gateway-server]]에 따르면 Pipecat pipeline은 per-connection이고 principal-to-session binding, origin allowlist, bearer/subprotocol auth, history projection, 그리고 한 session을 두고 두 socket이 경합할 때 처리하는 generation-based cancel-and-replace protocol에 대해 아무것도 주지 않습니다. 그 layer가 route를 소유하고 accept된 socket을 transport에 넘깁니다. Pipecat에는 그중 어떤 것에도 대응물이 없으므로, 그건 있던 자리에 그대로 있습니다.

### 13.4 스케치

실행 가능하지 않습니다 — transport는 [[ch-05/read]]의 주제입니다 — 하지만 아래 모든 줄은 위 표에서 나온 결정입니다.

```python
# --- startup, once per process ---
@app.on_event("startup")
async def _startup():
    app.state.resources = await build_process_resources()   # MCP, DB, HTTP — ONCE
    app.state.runner = WorkerRunner(handle_sigint=True, handle_sigterm=True)
    # hold a strong reference; see run.py:209-211
    app.state.runner_task = asyncio.create_task(
        app.state.runner.run(auto_end=False)                # §3
    )

# --- per call ---
@app.websocket("/ws/{session_id}")
async def _call(ws: WebSocket, session_id: str):
    await authorize(ws, session_id)                          # yours; §13.3.3
    transport = build_transport(ws)                          # ch-05
    worker = PipelineWorker(
        build_lina_pipeline(transport),                      # ch-11, ch-12
        name=f"call-{session_id}",                           # unique; §3.1a
        app_resources=app.state.resources,                   # §13.1.6
        idle_timeout_secs=45.0,                              # §9.1
        cancel_runner_on_idle_timeout=False,                 # §9.1 — mandatory
        setup_timeout_secs=5.0,                              # §6.1
    )
    await app.state.runner.add_workers(worker)               # starts immediately

    @transport.event_handler("on_client_connected")
    async def _connected(transport, client):
        await worker.queue_frames([LLMRunFrame()])           # §11 — Lina speaks first

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(transport, client):
        await worker.cancel(reason="customer hangup")        # NOT runner.cancel(); §3.2

    await worker.wait()                                      # or the drain watchdog
```

저 스케치에서 `06-voice-agent.py`를 복사했다면 틀렸을 줄이 셋입니다: `auto_end=False`, `cancel_runner_on_idle_timeout=False`, 그리고 `runner.cancel()` 대신 `worker.cancel()`. 각각이 작동하는 demo를 모든 동시 고객을 떨어뜨리는 host로 바꿉니다. 그게 README가 아니라 `runner.py`를 읽은 것의 가치입니다.

---

## 다음 챕터로

이 chapter가 앞으로 넘기는 것들, 이후 chapter가 재유도하지 않고 인용할 수 있도록 이름을 붙입니다:

- **`N/r` 산수**(§5.1) — control latency는 queue depth 나누기 drain rate이고, 유일한 자유 변수는 *N*입니다. [[ch-08/read]]은 이걸 써서 interruption cascade가 producer를 멈추는 것에 그치지 않고 output queue, audio clock, carrier의 playout buffer까지 tear down해야 하는 이유를 설명합니다. [[ch-11/read]]은 latency budget에서 다시 쓰는데, 거기서 `audio_out_10ms_chunks = 4`가 interrupt-granularity 항으로 재등장합니다.
- **processor당 two-queue / two-task model**(§4) — priority tier 1/10/20, system frame은 input task 위 inline, 나머지는 cancel 가능한 process task 위, `FrameQueue.reset()`은 `UninterruptibleFrame`을 유지. [[ch-06/read]]은 VAD에서 발생한 turn signal이 buffering된 audio보다 먼저 도착하는 이유를 설명하는 데 필요하고; [[ch-08/read]]은 cascade에 필요하고; [[ch-12/read]]은 여러분의 rule-layer processor가 `await` 도중에 cancel되어 자기 accumulator를 스스로 reset해야 하기 때문에 필요합니다.
- **bounded/unbounded 비대칭**(§7, §8) — `CancelFrame`은 20 s bound를 가진 `SystemFrame`이고, `EndFrame`은 bound가 전혀 없는 `ControlFrame, UninterruptibleFrame`이며, 이유는 한 경로는 지킬 게 없고 다른 경로는 오직 무언가를 지키려고 존재하기 때문입니다.
- **`worker.queue_frame`의 direction 분기**(§10) — downstream은 `_push_queue`를 통해 head로, 나머지는 `_sink`를 통해 tail로. [[ch-10/read]]의 `FlowManager` injection과 [[ch-12/read]]의 transition race는 둘 다 이것 없이는 읽히지 않습니다.
- **Lina host topology**(§13) — process 하나, `WorkerRunner(auto_end=False)` 하나, 통화당 `PipelineWorker` 하나, 모든 session은 같은 loop 위의 `asyncio.Task`, per-call shutdown은 `worker.*`로만, 그리고 framework가 제공하지 않는 이름 붙은 작업 셋.

[[ch-05/read]]은 이 chapter가 의도적으로 비워둔 한 자리를 채웁니다: canonical chain의 1번과 6번 위치인 `transport.input()`과 `transport.output()`. audio가 실제로 어디서 오는지 — WebRTC인지, raw websocket인지, 아니면 프로토콜 차이가 전부 `FrameSerializer` 하나로 환원되는 telephony carrier인지 — 에 답하고, boson의 `gateway/server/`에 나란히 놓을 구체적 Pipecat 대응물이 처음 생기는 chapter입니다.

나중을 위해 여기 park해 두는 열린 질문들, 잃어버리지 않도록:

- **container-per-call vs task-per-call.** §13은 Lina host에 대해 task-per-call을 골랐습니다. [[deployment-scaling]]은 반대 압력을 짚습니다: telephony는 webhook으로 도착하므로, container-per-session model은 동시 통화 peak에 맞춘 warm pool이 필요합니다. 그건 correctness가 아니라 cost/architecture trade이고, [[ch-13/read]]에 속합니다.
- **`processor_unusable_policy`.** `CONTINUE` / `END` / `CANCEL` (`worker.py:143-160`), processor당 한 번 적용. 선택에는 [[ch-05/read]]과 [[ch-07/read]]의 provider-failover 증거가 필요합니다.
- **boson의 tool_use/tool_result repair가 갈 자리.** [[boson-interrupt-subsystem]]에 따르면 `InterruptionFrame`은 turn을 truncate하지만 `ToolResultBlock`을 합성하지는 않습니다. §4.3은 process task가 `await` 도중에 죽는다고 말했고; repair는 여러분이 직접 쓰고 직접 reset하는 processor state입니다. [[ch-09/read]]이 그걸 소유합니다.
