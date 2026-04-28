---
calling_spec:
  purpose: Deep walkthrough of MCPClient, MCPManager, and mcp/bridge
  chapter: ch-06
  course: boson-agent
  phase: read
  excerpt_of: read.md
---

# MCP Integration — Deep Walkthrough

Three files implement MCP support: `client.py` (single-server connection),
`manager.py` (multi-server lifecycle), and `bridge.py` (protocol translation).
Together they convert an external stdio subprocess into a `ToolSpec` that the
router can dispatch like any native tool.

---

## `client.py` — Single Server Connection

```python
# packages/basement/basement/mcp/client.py, lines 23-86

class MCPClient:
    """Manages a connection to a single MCP server via stdio transport."""

    def __init__(self, name: str, config: MCPServerConfig) -> None:
        self.name = name
        self.config = config
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        """Spawn the MCP server process and establish a session."""
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env if self.config.env else None,
        )

        self._exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            logger.debug("Connected to MCP server '%s'", self.name)
        except OSError as exc:
            await self._cleanup()
            raise MCPError(f"Failed to start MCP server '{self.name}': {exc}") from exc
        except Exception as exc:
            await self._cleanup()
            raise MCPError(f"Failed to connect to MCP server '{self.name}': {exc}") from exc

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server and return the result as a string."""
        if self._session is None:
            raise MCPError(f"Not connected to MCP server '{self.name}'")

        try:
            result = await self._session.call_tool(tool_name, arguments)
        except Exception as exc:
            raise MCPError(f"Tool call '{tool_name}' on server '{self.name}' failed: {exc}") from exc

        if result.isError:
            content_text = _extract_text(result.content)
            raise MCPError(f"Tool '{tool_name}' returned an error: {content_text}")

        return _extract_text(result.content)
```

**Mechanical explanation.**

`connect()` uses `AsyncExitStack` to manage two nested async context managers in one
cleanup handle. `stdio_client(params)` spawns the subprocess and returns a
`(read_stream, write_stream)` pair. `ClientSession(read_stream, write_stream)` wraps
those streams in the MCP wire protocol. `session.initialize()` performs the MCP
handshake (capability negotiation). All three steps are guarded: if any raises, the
exit stack's `aclose()` cleans up whatever was already opened.

`call_tool()` delegates to `self._session.call_tool()` (the MCP SDK method). The SDK
returns a result object with an `isError` flag and a `content` list of typed blocks.
If `isError` is set, the client raises `MCPError` rather than returning the error text
silently — this propagates to `ToolRouter.dispatch()`'s exception handler, which wraps
it into a `ToolResultBlock(is_error=True)`.

**Notice:** `_extract_text(result.content)` iterates over MCP content blocks and
extracts `.text` or `.data` fields. This means MCP tools that return structured content
(images, binary data) are flattened to strings — the LLM sees plain text regardless of
the MCP server's native output type.

---

## `bridge.py` — Protocol Translation

```python
# packages/basement/basement/mcp/bridge.py, lines 18-45

def create_mcp_handler(client: "MCPClient", tool_name: str) -> Callable:
    """Create an async handler that delegates to client.call_tool()."""

    async def handler(**kwargs: Any) -> str:
        return await client.call_tool(tool_name, kwargs)

    handler.__name__ = f"mcp_handler_{tool_name}"
    return handler


def mcp_tool_to_spec(server_name: str, mcp_tool: dict, client: "MCPClient") -> ToolSpec:
    """Convert a raw MCP tool dict to a ToolSpec.

    The resulting ToolSpec name follows the convention: mcp:{server_name}:{tool_name}.
    """
    tool_name: str = mcp_tool["name"]
    description: str = mcp_tool.get("description", "")
    input_schema: dict = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})

    qualified_name = f"mcp:{server_name}:{tool_name}"
    handler = create_mcp_handler(client, tool_name)

    return ToolSpec(
        name=qualified_name,
        description=description,
        input_schema=input_schema,
        handler=handler,
    )
```

**Mechanical explanation.**

`create_mcp_handler` is a closure factory. Each call closes over a specific `client`
instance and a specific `tool_name` string, returning an `async def handler(**kwargs)`
that calls `client.call_tool(tool_name, kwargs)`. The `handler.__name__` is set
explicitly so that `inspect.iscoroutinefunction` and debug logging show a useful name
instead of the generic `"handler"`.

`mcp_tool_to_spec` takes the raw dict from `client.list_tools()` and builds a
`ToolSpec`. Crucially, it prefixes the name with `mcp:{server_name}:` — so a tool
named `query` on server `postgres` becomes `mcp:postgres:query` in the dispatch table.
This namespacing prevents collisions when multiple MCP servers expose tools with the
same bare name.

**Notice:** The `input_schema` is taken verbatim from the MCP server's response
(`tool.inputSchema`). MCP servers already publish JSON Schema for their tools —
no auto-generation is needed. This is the symmetric counterpart to [[ch-04]]'s `@tool`
decorator, which synthesises JSON Schema from Python type hints. Both paths produce a
`ToolSpec` with an `input_schema` field; the router is indifferent to which path was
used.

---

## `manager.py` — Multi-Server Lifecycle

```python
# packages/basement/basement/mcp/manager.py, lines 30-71

async def start_all(self) -> int:
    """Connect all enabled servers. Returns total number of tools discovered."""
    total_tools = 0

    for name, config in self._configs.items():
        if not config.enabled:
            logger.debug("Skipping disabled MCP server '%s'", name)
            continue

        client = MCPClient(name, config)
        try:
            await client.connect()
            self._clients[name] = client
            logger.info("Connected to MCP server '%s'", name)

            raw_tools = await client.list_tools()
            for raw_tool in raw_tools:
                spec = mcp_tool_to_spec(name, raw_tool, client)
                self._tools.append(spec)
                total_tools += 1

        except MCPError as exc:
            logger.error("Failed to start MCP server '%s': %s", name, exc)

    logger.info("MCP Manager started: %d tool(s) from %d server(s)", total_tools, len(self._clients))
    return total_tools

def get_all_tools(self) -> list[ToolSpec]:
    """Return merged list of ToolSpec from all connected servers."""
    return list(self._tools)
```

**Mechanical explanation.**

`start_all()` iterates the server config dict. For each enabled server it:
1. Creates an `MCPClient` instance
2. Calls `client.connect()` (spawns subprocess, handshakes)
3. Calls `client.list_tools()` (MCP capability discovery)
4. Converts each raw tool dict to a `ToolSpec` via `mcp_tool_to_spec`
5. Accumulates specs into `self._tools`

The `except MCPError` block is deliberately broad — if one server fails to start, the
others still connect. The manager logs the failure and continues. This is fail-partial
design: a misconfigured postgres server does not prevent a filesystem server from
connecting.

`get_all_tools()` returns a snapshot of `self._tools`. This is called once by
`ToolRouter.register_mcp(mcp_manager)` at startup; after that the router holds its own
copy in `_dispatch_table`. If a server disconnects and reconnects, the router would need
to re-register — there is no live sync between manager and router after initial wiring.

---

## Full MCP Call Path

```
LLM outputs: use_tool("mcp:postgres:query", {"sql": "SELECT ..."})
  → ToolRouter.dispatch("mcp:postgres:query", {"sql": "..."})
    → PermissionChecker.check_tool("mcp:postgres:query")  [global policy]
    → stage gate check                                     [per-stage policy]
    → dispatch_table["mcp:postgres:query"] = ToolSpec{handler=mcp_handler_query}
    → await mcp_handler_query(sql="SELECT ...")
      → MCPClient.call_tool("query", {"sql": "SELECT ..."})
        → self._session.call_tool("query", {"sql": "..."})  [MCP wire protocol]
        → _extract_text(result.content)
      ← "id | name\n1  | Alice"
    ← ToolResultBlock(content="id | name\n1  | Alice", is_error=False)
```

Every layer converts between representations: the LLM's tool-use JSON → router
dispatch → Python closure → MCP SDK call → stdio → external process → MCP response
→ string → `ToolResultBlock` → next LLM message.
