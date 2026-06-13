# DeepSeek Proxy

A FastAPI server that exposes DeepSeek and Qwen Chat behind three drop-in API surfaces — OpenAI, Anthropic, and the OpenAI Responses API (used by Codex CLI) — with full DSML tool-call support and streaming on every route.

## Adapter URLs

The first URL segment selects the upstream adapter:

- DeepSeek base URL: `http://{host}:{port}/ds`
- Qwen base URL: `http://{host}:{port}/qwen`

For example, OpenAI chat completions are available at:

- `POST http://localhost:8000/ds/v1/chat/completions`
- `POST http://localhost:8000/qwen/v1/chat/completions`

The adapter URL decides which service receives the request. The request's
`model` field controls model features within that adapter.

---

## File layout

```
.
├── app.py              # FastAPI application (this repo)
├── ds_api
├────── adaptor.py             # DeepSeekAdapter — session management, HTTP bridge
├────── tool_dsml.py           # DSML format parser / formatter / prompt builder
├────── tool_sieve.py          # StreamSieve — real-time DSML extraction from token stream
├────── sha3_wasm_bg.wasm      # SHA-3 WASM used by the adaptor for request signing
├────── __init__.py
├── .env                       # environment variables (see below)
└── .requirements.txt          # package list
```

---

## Environment variables

Create a `.env` or copy `.example.env` file in the same directory as `app.py`, or export them in your shell.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DEEPSEEK_TOKEN` | yes | — | Bearer token from `chat.deepseek.com` (grab from browser DevTools → Network → Authorization header) |
| `QWEN_TOKEN` | for Qwen | — | Bearer token from `chat.qwen.ai` |
| `QWEN_COOKIES` | for Qwen | — | Session cookies from `chat.qwen.ai` |
| `DEEPSEEK_COOKIES` | yes | — | Full cookie string from the same session |
| `API_KEY` | no | *(empty)* | If set, every request must carry `Authorization: Bearer <API_KEY>`. Leave empty to disable the guard entirely. |
| `WASM_PATH` | no | `sha3_wasm_bg.wasm` | Path to the SHA-3 WASM binary. Only needed if you move the file. |

---

## Installation

```bash
pip install fastapi uvicorn httpx python-dotenv wasmtime
```
or 

```bash
pip install -r requirements.txt
```
Python 3.11+ is recommended (uses `X | Y` union syntax and `list[T]` generics throughout).

---

## Running

```bash
# development
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# production
python app.py
```

Interactive docs are available at `http://localhost:8000/docs` once the server is running.

---

## API surfaces

### OpenAI — Chat Completions

`POST /{adapter}/v1/chat/completions`

Drop-in replacement for the OpenAI Chat Completions endpoint. Accepts the full OpenAI request shape and returns identical response envelopes.

```bash
curl http://localhost:8000/ds/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

With tool calls:

```bash
curl http://localhost:8000/ds/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "What is the weather in Mumbai?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

**Streaming** — set `"stream": true`. Response is standard SSE with `chat.completion.chunk` deltas, including a `tool_calls` delta chunk when tool calls are present.

**Thinking / reasoning models** — use a model name containing `r1`, `reasoner`, or `think` (e.g. `"model": "deepseek-r1"`) to route to DeepSeek's expert/chain-of-thought mode.

**Web search** — use a model name containing `search` or `online` (e.g. `"model": "deepseek-search"`) to enable DeepSeek's built-in web search.

---

### Anthropic — Messages API

`POST /{adapter}/v1/messages`

Drop-in replacement for the Anthropic Messages endpoint. Accepts Anthropic's request shape and returns `message` objects with `text` and `tool_use` content blocks.

```bash
curl http://localhost:8000/ds/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "max_tokens": 1024,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Explain async/await in Python."}]
  }'
```

With tools:

```bash
curl http://localhost:8000/ds/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Search for the latest Nifty price."}],
    "tools": [{
      "name": "web_search",
      "description": "Search the web",
      "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
      }
    }]
  }'
```

**Streaming** — set `"stream": true`. Events follow the Anthropic SSE format: `message_start`, `content_block_start`, `content_block_delta` (text or `input_json_delta`), `content_block_stop`, `message_delta`, `message_stop`.

**Using with the Anthropic Python SDK:**

```python
import anthropic

client = anthropic.Anthropic(
    api_key="any",                         # only checked if API_KEY is set server-side
    base_url="http://localhost:8000",
)

response = client.messages.create(
    model="deepseek-chat",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
```

---

### OpenAI Responses API

`POST /{adapter}/v1/responses`

The stateful Responses API introduced in the `openai` Python SDK v2 and used by **Codex CLI** (`openai` CLI tool). This is the primary surface for agentic tool-use loops.

```bash
curl http://localhost:8000/ds/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "input": "Refactor this function to use list comprehension.",
    "instructions": "You are a senior Python engineer."
  }'
```

Multi-turn input array (SDK v2 format):

```bash
curl http://localhost:8000/ds/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "input": [
      {"role": "user", "content": "What does os.walk do?"},
      {"role": "assistant", "content": "It recursively yields directory trees..."},
      {"role": "user", "content": "Show me an example."}
    ]
  }'
```

With `reasoning.effort` (forces thinking mode regardless of model name):

```bash
curl http://localhost:8000/ds/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "input": "Prove that sqrt(2) is irrational.",
    "reasoning": {"effort": "high"}
  }'
```

**Streaming** — set `"stream": true`. Events follow the Responses API SSE format: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta` (one per token), `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.completed`.

Thinking tokens from DeepSeek expert mode are surfaced as `response.reasoning_summary_text.delta` events so Codex CLI can render them as a collapsible reasoning block.

**Using with Codex CLI:**

```bash
OPENAI_API_KEY=any \
OPENAI_BASE_URL=http://localhost:8000 \
codex "Explain this codebase"
```

**Using with the openai Python SDK v2:**

```python
from openai import OpenAI

client = OpenAI(api_key="any", base_url="http://localhost:8000")

response = client.responses.create(
    model="deepseek-chat",
    input="Write a binary search in Python.",
)
print(response.output[0].content[0].text)
```

---

### Legacy Codex completions

`POST /{adapter}/v1/completions`
`POST /{adapter}/v1/engines/{engine}/completions`

For tooling that still uses the pre-chat completions format (older versions of GitHub Copilot plugins, direct Codex API integrations).

```bash
curl http://localhost:8000/ds/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "code-davinci-002",
    "prompt": "def fibonacci(n):",
    "max_tokens": 200,
    "stream": false
  }'
```

The classic engine path:

```bash
curl http://localhost:8000/ds/v1/engines/code-davinci-002/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "# Sort a list\n", "max_tokens": 100}'
```

Supports `echo` (prepend prompt to output) and `suffix` (append to output), matching the original Codex API behaviour.

---

### Utility endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/{adapter}/v1/models` | List model IDs for the selected adapter |
| `GET` | `/{adapter}/v1/models/{id}` | Get a model record for the selected adapter |
| `GET` | `/{adapter}/health` | Liveness check for the selected adapter |

---

## Model routing

The server maps model name keywords to DeepSeek backend modes automatically. You never need to change the server config — just change the model string in your request.

| Model name contains | DeepSeek mode |
|---|---|
| `r1`, `reasoner`, `think` | Expert / chain-of-thought (`thinking_enabled=True`) |
| `search`, `online` | Web search enabled |
| anything else | Standard chat |

`reasoning.effort` in a Responses API request (`"medium"` or `"high"`) also forces thinking mode, independently of the model name.

---

## Tool calls (DSML)

DeepSeek does not natively support the OpenAI function-calling wire protocol. This proxy bridges the gap using DSML (DeepSeek Markup Language):

1. `build_dsml_tool_prompt()` appends a structured XML calling convention to the system message before the request is sent.
2. DeepSeek responds with `<|DSML|tool_calls>` XML when it wants to call a tool.
3. `StreamSieve` intercepts these tags in real time as tokens arrive, splitting them from plain text.
4. `parse_dsml_tool_calls()` parses the XML into standard OpenAI `tool_calls` dicts.
5. The route handler re-emits them in the target API format (OpenAI `tool_calls` field, Anthropic `tool_use` blocks, Responses API `function_call` output items).

This is entirely transparent to the calling SDK — it sees the same tool-call envelope it would from a native API.

---

## Architecture

```
Client
  │
  ▼
FastAPI route handler
  │  normalise request → messages list
  │  inject DSML tool prompt into system message
  │  _prepare_session() / _run_sync()
  ▼
DeepSeekAdapter.create_session()
DeepSeekAdapter.chat() or .chat_stream()    ← blocking; runs in thread pool
  │
  ▼  (streaming path)
asyncio.Queue  ←  producer thread
  │
  ▼
StreamSieve.feed(token)
  │  text events  →  forwarded as SSE deltas
  │  tool_calls events  →  buffered, emitted at end
  ▼
parse_dsml_tool_calls()
  │
  ▼
Translated SSE envelope  →  Client
```

Blocking I/O from `DeepSeekAdapter` is isolated in a `run_in_executor` thread pool so the `asyncio` event loop stays unblocked during long streaming responses.

---

## Authentication

If `API_KEY` is set in the environment, every request must include:

```
Authorization: Bearer <your-API_KEY>
```

Leave `API_KEY` empty (or unset) to disable the guard for local use.
