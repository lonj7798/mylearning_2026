here is what I learned

from the basement code, mostly deep dive into the `decorator`
from boson-agent there are 3 main decorators; `@tool`, `@hook`, `@check` (the Gateway rule decorator is `@check`, not `@rule` — "rule" is the concept, `@check` is the actual marker in `rules/engine.py`)

and those decorator help to register but also easy to wrap up into usable format. 

decorator itself can be used as a type of looger too, but the most important functionality is `hook`
`hook` can modify the middle of the code or change the state whenever I need to check some points. 

the mofrmat of decorator is a nested fuctions. the reaoson of nested function is to block decorator call the input function direcrtly. 
so the overal shape is like

```
def count_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{end-start}")
        return result            # ← preserve the original return value
    return wrapper

@count_time
def sum(a, b):
    return a+b
```

and decorator itself can take variable too. for example, if I wnat to repeat some function N times, 

```
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def sum(a, b):
    return a+b
```


I think this is the very basic knowledge to understand the design choice of `boson-agent`

my question was why most of the function inside of single lopp included `async` instead of `sync`.
three reasons stacked:

1. **Single Gateway, many sessions.** Multi-session concurrency on one event loop — one session's I/O wait can't block other sessions.
2. **LLM streaming itself is async.** Providers return `AsyncIterator[StreamEvent]`; consuming requires `async for`. Even a single-session CLI agent needs async because of this.
3. **Async is transitive.** Once `run_agent_loop` is async (forced by streaming), every helper it calls — `fire_event`, hook handlers, tool handlers — must also be async, or the chain breaks.

except that one, there is so many design choice like meta-tool, permission, skill, ...
I think we need to check the purpose of meta-tool here. 
there is a clear trade-off on tool exposure:

- **Native tool calling** — every allowed tool is in the LLM API request. LLM cannot attempt a tool that wasn't sent. No wasted attempts, but the tool list is baked into the request and less flexible at runtime.
- **Meta-tool (`use_tool` / `use_skill`)** — LLM only sees two meta-tools. The actual catalog is managed by `ToolRouter`. Pros: fewer schema tokens, easy stage-based swapping via `set_allowed_tools()`. Cons: LLM may try to call a disallowed tool (it knows meta exists, doesn't know the live allow-list) and the router rejects *after* the attempt — wasted tokens.
- Both paths can still be systematically limited by `PermissionChecker` (allow/deny lists, deny > allow) which sits below either calling style.


from gateway, it is also based on `async` and also.  handle some amount of interruption, but didn't dig it that much. instead focusing on how overal gateway works. 
manage the session by session, depends on the result of `rules` from gateway, it can modify the input to Agent (I mean user message) like adding hook, change stage, pre-load tool/skills. 
and tools, hooks, skills are based on the folder system (Basement discovers `tools/*.py`, `hooks/*.py`, `skills/*.md` from the agent folder). **Rules are different** — they live in Gateway, registered via the `@check` decorator, and wired into `RuleEngine` by the agent's `config.py`, not auto-scanned from a folder.

also from tool calling side, once flag 'meta-tool' then input is change. before that it support native tool callings. 
