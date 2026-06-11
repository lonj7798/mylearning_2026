<!-- chapter: ch-03
     track: internals
     kind: content
     title: Tool Discovery — Search + a From-Scratch BM25
     deps: [ch-02]
     sources: [[automationbench-harness]]
     figures: figures/bm25-explorer.html
-->

# 03장 — Tool Discovery: Search + a From-Scratch BM25

> **핵심 통찰.** Benchmark가 테스트하는 것은 모델이 ~400개 후보 중에서 올바른 tool을 *찾을 수 있는지* 여부다 — tool이 건네지고 나서 호출하는 능력이 아니다. Discovery가 첫 번째 인지적 난관이며, 이것을 제대로 처리해야 다른 모든 capability가 그 위에 쌓인다.

> **가이드라인.** Construction time에 tool corpus를 index하라(`name + docstring + param descriptions`에 BM25 적용). Agent에게 `search_tools` 함수 하나를 노출하고, token economy가 compression을 주도하게 하라 — verbose schema는 actionable한 동안에만 살아남는다.

---

## 1  Why Discovery Is a First-Class Tested Capability

AutomationBench의 headline toolset mode(`zapier`)는 모델에게 정확히 **두** 개의 tool을 준다: `search_tools`와 `execute_tool`. System prompt에 400개 이름이 미리 열거된 목록도 없고, 모델이 스캔할 수 있는 service directory도 없다. Gmail을 보내고 싶다면 먼저 올바른 tool 이름을 *search*해야 한다.

이것은 의도적인 설계다. 실제 automation agent는 동일한 제약에 직면한다. 엔터프라이즈 SaaS는 수백 개의 action을 노출하고, 모든 schema를 context window에 밀어 넣는 것은 비용상 불가능하다. Benchmark는 corpus를 숨기고 모델이 search를 통해 올바른 tool에 도달하는지를 측정함으로써 그 비용을 operationalize한다.

`limited_zapier` mode는 동일한 아이디어를 ablation으로 뒤집는다. 모델은 task별로 정확한 tool subset을 tool list에서 받는다 — discovery는 변수에서 제거된다. `zapier` vs `limited_zapier` 점수를 비교하면 *discovery tax*가 드러난다: tool을 찾아야 하는 것과 건네받는 것의 차이에서 비롯되는 accuracy 격차.

Test suite는 unit-test 관점에서 같은 요점을 전달한다. `tests/test_bm25.py`와 `tests/test_api_search.py`는 domain 및 rubric test와 나란히 first-class test file로 존재한다 — 나중에 생각한 것이 아니다. Discovery infrastructure는 production code다.

---

## 2  The From-Scratch BM25

Harness는 자체 BM25 구현을 `automationbench/utils/bm25.py`에 포함한다. 외부 ranking 라이브러리는 없고, 파일은 순수 Python 70줄이다.

### 2.1  Tokenizer

```python
# automationbench/utils/bm25.py  L10-12
def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, treating underscores as word separators."""
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
```

underscore replace 단계는 load-bearing이다. Tool 이름은 `{service}_{action}` 규칙을 따른다 — `gmail_send_email`, `salesforce_find_records`, `slack_post_message`. Underscore를 그대로 두면 tokenizer가 `gmail_send_email` 하나를 단일 token으로 내보내, `gmail` 같은 query 단어와 매칭되지 않는다. Replace를 거치면 이름이 regex 실행 전에 `"gmail send email"`로 바뀌고, 각 단어가 독립적인 index token이 된다. 테스트가 이를 명시적으로 확인한다:

```python
# tests/test_bm25.py  L129-133
def test_underscore_terms_in_docs(self):
    docs = ["slack_chat_post_message send message to channel"]
    scorer = BM25Scorer(docs)
    scores = scorer.scores("slack chat post")
    assert scores[0] > 0
```

### 2.2  BM25Scorer construction

```python
# automationbench/utils/bm25.py  L27-45
def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
    self.k1 = k1
    self.b = b

    self._tokenized: list[list[str]] = [tokenize(doc) for doc in docs]
    n = len(self._tokenized)
    self._avgdl = sum(len(d) for d in self._tokenized) / n if n else 1.0

    # Document frequency per term
    df: dict[str, int] = {}
    for doc in self._tokenized:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    # IDF with Robertson-Sparck Jones smoothing
    self._idf: dict[str, float] = {
        term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
        for term, freq in df.items()
    }
```

IDF 공식은 Robertson–Sparck Jones smoothed variant다: `log((N - df + 0.5) / (df + 0.5) + 1)`. 외부 log 앞의 `+ 1`은 corpus의 절반 이상에 등장하는 term에서 IDF가 음수가 되는 것을 막는다 — tool search에서는 흔한 상황인데, `gmail` 같은 service 이름이 같은 service의 여러 tool description에 걸쳐 등장하기 때문이다. Smoothing이 없으면 `"gmail"` 쿼리는 Gmail tool이 더 많이 index될수록 *낮은* 점수를 받게 된다.

파라미터 `k1=1.5`와 `b=0.75`는 BM25의 사실상 표준 default다. `k1`은 term-frequency saturation을 제어한다: document에서 어떤 term을 여러 번 반복해도 점수 이득이 비례적으로 늘지는 않는다. `b=0.75`는 length normalization을 적용한다 — 관련 단어 하나가 있는 긴 docstring은 짧고 집중된 설명보다 낮은 점수를 받는다. L119-127의 테스트는 기본값이 아닌 `k1`/`b` 값이 서로 다른 상대적 순서를 만드는지를 검증하며, 이것이 지켜야 할 올바른 동작이다.

### 2.3  Scoring and top_k

```python
# automationbench/utils/bm25.py  L47-69
def scores(self, query: str) -> list[float]:
    """Return a BM25 score for each document in the corpus."""
    terms = tokenize(query)
    result = []
    for doc_tokens in self._tokenized:
        dl = len(doc_tokens)
        score = 0.0
        for term in terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            tf = doc_tokens.count(term)
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            )
        result.append(score)
    return result

def top_k(self, query: str, k: int = 10) -> list[int]:
    """Return indices of the top-k scoring documents (descending order)."""
    scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [i for _, i in scored[:k]]
```

`top_k`는 slicing 전에 `score > 0`인 것만 필터링한다 — 겹치는 부분이 없는 document는 아예 제외되므로, sparse query에 대한 `top_k` 호출은 zero-score noise로 채우는 대신 `k`보다 적은 결과를 반환한다. Agent 입장에서 중요한 점: 빈 결과 목록은 다른 query를 시도하라는 명확한 신호인 반면, zero-score 결과 목록은 오해를 일으킨다.

이 scorer의 인터랙티브 버전 — 12개 tool docstring의 toy corpus에서 브라우저 위에서 live로 실행 — 이 **[figures/bm25-explorer.html](figures/bm25-explorer.html)**에 있다. Query를 입력하면 term별 점수 기여가 실시간으로 업데이트된다.

---

## 3  ToolRegistry and search_tools

BM25Scorer는 `automationbench/tools/zapier/meta.py`의 `ToolRegistry`에 연결된다. Registry는 module-level lazy singleton으로, `ALL_TOOLS`로부터 최초 사용 시 한 번 빌드된다.

### 3.1  What gets indexed

```python
# automationbench/tools/zapier/meta.py  L46-56
# Include parameter names + descriptions in searchable text (mirrors api_search index)
param_parts: list[str] = []
for param_name, param_info in params.get("properties", {}).items():
    param_parts.append(param_name)
    if isinstance(param_info, dict) and param_info.get("description"):
        param_parts.append(param_info["description"])
params_text = " ".join(param_parts)
searchable = f"{name}: {full_desc}"
if params_text:
    searchable += f" {params_text}"
self._searchable_texts.append(searchable)
```

각 tool의 index document는 `"name: docstring param param_desc ..."` 형태를 갖는다. Parameter 이름과 설명을 포함하는 것이 중요한 이유는, agent가 `"send email"`(action verb)이 아닌 `"subject body"`(field name)로 query할 수도 있기 때문이다. 두 경로 모두 `gmail_send_email`을 찾을 수 있어야 한다.

`world` 파라미터는 indexing 전에 설명에서 제거된다(L60-81): registry가 `_get_full_description`을 호출하며, 이 함수는 `world:`로 시작하는 행과 그 continuation indent를 제거한다. `world` 주입은 모델과 search index 모두에게 숨겨진 구현 세부사항이다.

### 3.2  search_tools and make_search_tools

```python
# automationbench/tools/zapier/meta.py  L134-154
def search_tools(query: str, top_k: int = 5) -> str:
    """Find available tools by name or description.

    Tool names follow the pattern {service}_{action} (e.g., salesforce_query,
    gmail_send_email, slack_send_channel_message).

    Uses BM25 keyword-based relevance search. Works with service names,
    action words, or multi-word queries.
    Examples: "salesforce", "send email", "update deal", "slack channel"

    Args:
        query: Search query — service names, keywords, or a description.
        top_k: Maximum number of results to return (default: 20).

    Returns:
        JSON string with a list of matching tools, each containing name,
        description, and parameter schema.
    """
    registry = _get_registry()
    results = registry.bm25(query, top_k=top_k)
    return json.dumps(results, indent=2)
```

함수 signature의 default `top_k=5`는 적정한 예산이다: query가 불명확할 때를 커버할 만큼 충분하지만(올바른 tool이 1위가 아닐 수 있다), 5개의 전체 JSON schema로 context window를 채우지 않을 만큼 적다. 각 결과는 세 key를 가진 dict다: `name`, `description`(정리된 docstring), 그리고 `parameters`(property 타입과 설명이 있는 전체 JSON schema). 이 schema가 모델이 `execute_tool`에 넘길 `arguments` JSON string을 구성하는 데 사용된다.

`make_search_tools`(L157-186)는 configurable default와 hard cap을 가진 동일한 함수의 변형을 반환한다:

```python
# automationbench/tools/zapier/meta.py  L157-186
def make_search_tools(default_top_k: int = 20, max_top_k: int | None = None) -> Callable:
    """Return a search_tools function with a custom default and optional hard cap on top_k."""
    cap = max_top_k
    default = default_top_k

    def _search_tools(query: str, top_k: int = default) -> str:
        ...
        actual_k = min(top_k, cap) if cap is not None else top_k
        results = registry.bm25(query, top_k=actual_k)
        return json.dumps(results, indent=2)

    _search_tools.__name__ = "search_tools"
    return _search_tools
```

Cap은 통제된 실험을 위해 중요하다. Harness가 `search_top_k=N`으로 초기화될 때(runner.py L83), `make_search_tools(max_top_k=N)`이 사용되며, 이는 모델이 임의로 큰 `top_k`를 요청해 전체 corpus를 스캔함으로써 discovery challenge를 우회하는 것을 막는다.

---

## 4  execute_tool

`search_tools`가 후보를 찾아내면 `execute_tool`이 호출을 dispatch한다:

```python
# automationbench/tools/zapier/meta.py  L189-204
def execute_tool(world: WorldState, tool_name: str, arguments: str) -> str:
    """Execute a discovered tool by name with the given arguments.

    Use search_tools first to find the right tool and its parameter schema,
    then call this with the tool name and a JSON string of arguments.

    Args:
        world: The current world state (injected automatically).
        tool_name: The exact tool name from search results.
        arguments: JSON string of arguments matching the tool's parameter schema.

    Returns:
        The tool's return value (JSON string).
    """
    registry = _get_registry()
    return registry.execute(tool_name, arguments, world=world)
```

내부적으로 `registry.execute`(L98-110)는 세 가지를 한다: 이름으로 함수를 조회하고, `arguments` string을 `json.loads`하고, 주입된 `world` kwarg를 병합한 뒤 함수를 호출한다. `tool_name`을 알 수 없으면 `ValueError`를 발생시키며, 그 메시지는 모델에게 명시적으로 `search_tools`를 사용하라고 알린다 — discovery로 다시 돌아가는 루프를 닫는다.

```python
# automationbench/tools/zapier/meta.py  L98-110
def execute(self, tool_name: str, arguments: str, **injected: Any) -> str:
    """Execute a tool by name with JSON arguments string."""
    func = self._tool_map.get(tool_name)
    if func is None:
        raise ValueError(
            f"Unknown tool: {tool_name}. Use search_tools to discover available tools."
        )
    parsed_args = json.loads(arguments)
    merged = {**parsed_args, **injected}
    result = func(**merged)
    if isinstance(result, str):
        return result
    return json.dumps(result)
```

---

## 5  API-Mode Discovery: api_search

`api` toolset mode는 모델에게 Zapier meta-tool 대신 `api_search`와 `api_fetch`를 준다. `api_search`(`automationbench/tools/api/search.py`)는 REST endpoint의 flat tab-separated index에 대해 BM25를 실행한다:

```python
# automationbench/tools/api/search.py  L83-95
def _build_index_line(api_name: str, endpoint: dict) -> str:
    """Build one tab-separated searchable line for an endpoint.

    Format: api_name<TAB>endpoint_id<TAB>method<TAB>path<TAB>searchable_text
    searchable_text includes the endpoint description plus all parameter descriptions.
    """
    desc_parts = [endpoint.get("description", "")]
    for param_info in endpoint.get("parameters", {}).values():
        if isinstance(param_info, dict) and param_info.get("description"):
            desc_parts.append(param_info["description"])
    searchable = " ".join(filter(None, desc_parts))
    fields = [api_name, endpoint["id"], endpoint["method"], endpoint["path"], searchable]
    return "\t".join(fields)
```

Index(`schemas/index.txt`)는 디스크의 index보다 새로운 `.jsonc` schema 파일이 하나라도 있을 때 lazy하게 재빌드된다(L107-115). 이것은 mtime 기반 incremental build다: 새 API schema를 추가하면 import 시가 아니라 다음 `api_search` 호출 시 한 번 재생성이 trigger된다.

```python
# automationbench/tools/api/search.py  L107-115
def _ensure_index(schemas: dict[str, dict]) -> list[str]:
    """Return index lines, regenerating index.txt if any schema file is newer."""
    schema_files = list(SCHEMAS_DIR.glob("*.jsonc"))
    needs_regen = not INDEX_FILE.exists() or any(
        f.stat().st_mtime > INDEX_FILE.stat().st_mtime for f in schema_files
    )
    if needs_regen:
        _regenerate_index(schemas)
    return INDEX_FILE.read_text().splitlines()
```

결과 형식은 `search_tools`와 다르다: 각 결과에는 Python 함수 이름 대신 `api_fetch`에 바로 넘길 수 있도록 준비된 resolved `url` field(base URL + stripped internal prefix)가 포함된다. `tests/test_api_search.py L107-111`의 `test_results_no_path_field` 테스트가 이 contract를 보호한다 — `path`는 없어야 하고 `url`은 있어야 한다. 모델이 internal routing prefix를 추론하게 해서는 안 되기 때문이다.

---

## 6  Token Economy: _compress_meta_messages

단순한 구현이라면 verbose search 결과가 대화 history에 무한정 쌓인다. 각 `search_tools` 결과는 전체 schema의 JSON array — 이름, docstring, type과 설명이 있는 모든 parameter. 10턴 episode에서 여러 번 search하면 수천 token의 dead weight로 불어난다.

`runner.py`의 `_compress_meta_messages`(L234)는 stale 결과가 더 이상 actionable하지 않으면 덮어써서 이를 해결한다:

```python
# automationbench/runner.py  L234-293
def _compress_meta_messages(
    self,
    messages: vf.Messages,
    tool_messages: vf.Messages,
    state: vf.State,
) -> vf.Messages:
    """Compress old search_tools results after execute_tool is called.

    Once the model acts on search results by calling execute_tool, the verbose
    search results (full descriptions + parameter schemas) are dead weight.
    Replace them with a brief tool name list to save tokens on future turns.

    Only compresses search results from PREVIOUS turns, never the current turn.
    This ensures schemas remain available for tools searched in the same turn
    as an execute_tool call, preventing argument-name hallucination when the
    model searches and executes in parallel.
    """
```

로직은 어떤 `search_tools` call ID가 이미 봤는지 추적하고, `execute_tool` 호출이 포함된 턴에서는 이전 search 결과 message를 `[Previously found: name1, name2, ...]`로 덮어쓴다. 핵심 불변식은 **현재 턴 제외**다: `execute_tool` 호출과 같은 턴에 나온 search 결과는 그대로 유지된다.

왜인가? 모델은 단일 parallel tool-call batch에서 search와 즉시 execute를 함께 할 수 있다. Execute 호출이 resolve되기 전에 search 결과가 압축되면, 모델은 방금 받아온 parameter schema 없이 `arguments`를 구성하게 된다 — argument hallucination이 보장된다. 구현은 set 연산으로 이를 추적한다:

```python
# automationbench/runner.py  L258-275  (key excerpt)
current_search_ids: set[str] = set()
has_execute = False
for tc in tool_calls:
    if tc.name == "search_tools":
        current_search_ids.add(tc.id)
    elif tc.name == "execute_tool":
        has_execute = True

# Accumulate this turn's search IDs for potential compression next turn
state.setdefault("_search_call_ids", set()).update(current_search_ids)

if not has_execute:
    return tool_messages

# Only compress searches from PREVIOUS turns (exclude current turn)
compressible_ids = state["_search_call_ids"] - current_search_ids
```

`state["_search_call_ids"]`는 지금까지 본 모든 search ID를 누적한다. `compressible_ids`는 누적된 set에서 현재 턴의 ID를 뺀 것이다. 그것들만 덮어쓴다. 결과 message는 다음과 같다:

```
[Previously found: gmail_send_email, gmail_list_messages, ...]
```

이것으로 모델은 이미 검색했다는 사실과 무엇을 찾았는지를 기억할 수 있으며, 전체 schema token 비용을 다시 지불하지 않아도 된다.

---

## 7  Test Suite as Executable Specification

Test suite는 discovery를 구현 세부사항이 아닌 first-class capability로 다룬다. `test_idf_computed` 테스트(L91-95)는 RSJ IDF 공식에 대한 특히 깔끔한 executable spec이다:

```python
# tests/test_bm25.py  L91-95
def test_idf_computed(self):
    scorer = BM25Scorer(["email send", "email draft", "calendar event"])
    # "email" appears in 2/3 docs, "calendar" in 1/3
    # IDF for rarer term should be higher
    assert scorer._idf["calendar"] > scorer._idf["email"]
```

이것이 코드로 표현된 IDF contract다: 희귀한 term이 더 높이 rank된다. 이를 위반하는 IDF 공식 변경 — 예를 들어 실수로 `+1` smoothing term을 빼먹는 것 — 은 이 assertion을 뒤집을 것이다.

`api_search`의 경우 deduplication 테스트가 가장 운영상 중요하다:

```python
# tests/test_api_search.py  L113-116
def test_deduplicates_results(self):
    result = json.loads(api_search("email", top_k=10))
    ids = [r["id"] for r in result["results"]]
    assert len(ids) == len(set(ids)), "Duplicate endpoint IDs in results"
```

`api_search`는 deduplication을 감안해 `top_k * 3` 개의 후보를 가져오기 때문에(L153), 같은 endpoint가 여러 index 행에 나타날 수 있다. `seen` set 없이는 agent가 `top_k=5` 결과에서 같은 endpoint를 두 번 받을 수 있고, result slot이 낭비된다.

---

## 8  Summary: The Discovery Stack

Agent query에서 tool 실행까지의 전체 discovery 경로:

```
agent emits: search_tools("send slack message", top_k=5)
             │
             ▼
ToolRegistry.bm25(query, top_k=5)
             │  tokenize("send slack message") → ["send","slack","message"]
             │  BM25Scorer.top_k(query, k=5)
             │    per-doc: score = Σ IDF(t) * TF_norm(t, doc)
             │    filter score>0, sort descending, return indices[:5]
             ▼
[{name: "slack_post_message", description: "...", parameters: {channel: ..., text: ...}}, ...]
             │
             ▼
agent emits: execute_tool("slack_post_message", '{"channel": "C123", "text": "Hello"}')
             │
             ▼
registry.execute → json.loads(arguments) + inject world → func(**merged) → result string
             │
             ▼  (next turn — if execute_tool was called)
_compress_meta_messages: previous search results → "[Previously found: slack_post_message, ...]"
```

세 가지 toolset mode(`zapier`, `limited_zapier`, `api`)는 각각 이 stack의 다른 부분을 테스트하며, benchmark가 accuracy 차이를 discovery vs. execution vs. REST-shape reasoning으로 독립적으로 귀속시킬 수 있게 한다.

전체 episode lifecycle과 `limited_zapier` ablation 설계는 [[automationbench-harness]]를 참고하라.
