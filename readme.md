# DS API Proxy

DS API Proxy is a FastAPI service that exposes OpenAI-, Anthropic-, and
Codex-compatible endpoints over authenticated DeepSeek, Qwen, and ChatGPT web
sessions.

It provides a common local API for chat, streaming, legacy text completions,
and function calling. Function definitions are translated through DSML
(DeepSeek Markup Language), allowing clients to receive familiar OpenAI or
Anthropic tool-call responses.

> [!IMPORTANT]
> This project uses internal, undocumented web endpoints and browser-session
> credentials. It is not an official SDK for DeepSeek, Qwen, OpenAI, or
> Anthropic. Upstream changes may break compatibility without notice. Use only
> accounts and credentials you are authorized to access.

## Features

- DeepSeek, Qwen, and ChatGPT upstream adapters
- OpenAI Chat Completions-compatible endpoint
- Anthropic Messages-compatible endpoint
- OpenAI Responses-compatible endpoint for agent and Codex-style clients
- Legacy OpenAI text completion endpoints
- Server-Sent Events (SSE) streaming
- Function-tool translation through DSML
- DeepSeek reasoning and web-search routing by model name
- Optional bearer-token protection
- FastAPI OpenAPI documentation

## Architecture

```text
Client
  |
  | OpenAI / Anthropic / Responses request
  v
FastAPI route
  |
  | Normalize messages and inject DSML tool instructions
  v
Selected adapter: /ds, /qwen, or /chatgpt
  |
  | Authenticated request to the provider's web backend
  v
Upstream token stream
  |
  | StreamSieve separates text from DSML tool calls
  v
Translated JSON response or SSE stream
```

Blocking upstream HTTP work runs in an executor so long-running requests do
not directly block FastAPI's event loop.

## Repository Layout

```text
.
|-- app.py                     FastAPI application and protocol translation
|-- ds_api/
|   |-- adaptor.py             DeepSeek, Qwen, and ChatGPT adapters
|   |-- tool_dsml.py           DSML prompt builder, parser, and formatter
|   |-- tool_sieve.py          Incremental DSML extraction for streams
|   |-- sha3_wasm_bg.wasm      DeepSeek proof-of-work helper
|   `-- __init__.py
|-- requirements.txt           Python dependencies
|-- .example.env               Environment-variable template
|-- codex_config.toml          Example Codex provider configuration
|-- setting.json               Example Anthropic-compatible client settings
|-- run.bat.example            Windows launch template
|-- run.sh.example             Shell launch template
|-- example.py                 Direct Qwen HTTP experiment
`-- example2.py                Direct QwenAdapter experiment
```

## Requirements

- Python 3.10 or newer
- An authenticated browser session for at least one supported upstream
- The bundled `ds_api/sha3_wasm_bg.wasm` file

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .example.env .env
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .example.env .env
```

Edit `.env` and configure only the adapters you intend to use.

## Configuration

| Variable | Required | Description |
|---|---:|---|
| `DEEPSEEK_COOKIES` | For DeepSeek | Cookie header from an authenticated `chat.deepseek.com` session |
| `DEEPSEEK_TOKEN` | For DeepSeek | Bearer token used by the DeepSeek web application |
| `QWEN_COOKIES` | For Qwen | Cookies from an authenticated `chat.qwen.ai` session |
| `QWEN_TOKEN` | Usually for Qwen | Bearer token used by the Qwen web application |
| `CHATGPT_COOKIES` | For ChatGPT | Cookie header from an authenticated `chatgpt.com` session |
| `CHATGPT_COOKIE_PART_1` | Optional | First half of a split ChatGPT cookie value |
| `CHATGPT_COOKIE_PART_2` | Optional | Second half of a split ChatGPT cookie value |
| `CHATGPT_TOKEN` | Optional | ChatGPT bearer token; otherwise one is obtained from session cookies |
| `API_KEY` | Optional | Bearer token required from proxy clients; empty disables authentication |
| `WASM_PATH` | Optional | DeepSeek WASM path; defaults to `ds_api/sha3_wasm_bg.wasm` |

An adapter is initialized only when its cookie variable is present. ChatGPT
accepts either `CHATGPT_COOKIES` or the concatenated values of
`CHATGPT_COOKIE_PART_1` and `CHATGPT_COOKIE_PART_2`.

The application reads `API_KEY` (singular). When configured, protected
endpoints require:

```http
Authorization: Bearer <API_KEY>
```

Do not commit `.env`, browser cookies, access tokens, or copied authorization
headers. The included `.gitignore` excludes `.env`.

## Running the Server

For local development:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

To listen on all network interfaces:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

You can also run `python app.py`. The direct Python entry point listens on
`0.0.0.0:8000`.

Once running:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`
- DeepSeek health: `http://localhost:8000/ds/health`
- Qwen health: `http://localhost:8000/qwen/health`
- ChatGPT health: `http://localhost:8000/chatgpt/health`

A health response reports `adapter_ready: true` only when the adapter was
configured and its startup session was created successfully.

## Adapter Base URLs

| Adapter | Base URL |
|---|---|
| DeepSeek | `http://localhost:8000/ds/v1` |
| Qwen | `http://localhost:8000/qwen/v1` |
| ChatGPT | `http://localhost:8000/chatgpt/v1` |

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/{adapter}/v1/chat/completions` | OpenAI-style chat completions |
| `POST` | `/{adapter}/v1/messages` | Anthropic-style messages |
| `POST` | `/{adapter}/v1/responses` | OpenAI Responses-style requests |
| `POST` | `/{adapter}/v1/completions` | Legacy text completions |
| `POST` | `/{adapter}/v1/engines/{engine}/completions` | Classic engine-based completions |
| `GET` | `/{adapter}/v1/models` | List advertised model identifiers |
| `GET` | `/{adapter}/v1/models/{model_id}` | Retrieve an advertised model |
| `GET` | `/{adapter}/health` | Check adapter readiness |

Replace `{adapter}` with `ds`, `qwen`, or `chatgpt`.

## Quick Start

### OpenAI Chat Completions

```bash
curl http://localhost:8000/ds/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "Explain dependency injection briefly."}
    ]
  }'
```

Set `"stream": true` to receive `chat.completion.chunk` SSE events.

### Anthropic Messages

```bash
curl http://localhost:8000/qwen/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{
    "model": "qwen3.7-plus",
    "max_tokens": 1024,
    "system": "You are a concise technical assistant.",
    "messages": [
      {"role": "user", "content": "What is an event loop?"}
    ]
  }'
```

Set `"stream": true` to receive Anthropic-style message and content-block
events.

### OpenAI Responses

```bash
curl http://localhost:8000/chatgpt/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{
    "model": "auto",
    "instructions": "You are a senior Python engineer.",
    "input": "Write a small retry decorator."
  }'
```

The Responses route accepts a string or a message array in `input`. With
`"stream": true`, it emits `response.*` SSE events and concludes with
`data: [DONE]`.

If `API_KEY` is not configured, omit the `Authorization` header.

## Function Tools

OpenAI-style function tools can be sent to the Chat Completions endpoint:

```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "user", "content": "What is the weather in Mumbai?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

The proxy converts tool definitions into a DSML instruction, detects DSML
tool-call markup in the response, and returns OpenAI `tool_calls`, Anthropic
`tool_use`, or Responses `function_call` items.

Tool execution remains the client's responsibility. Only function tools are
translated by the Responses endpoint; other tool types are currently accepted
as no-op compatibility shims.

## Model Routing

For DeepSeek, model-name keywords control upstream features:

| Model name contains | Behavior |
|---|---|
| `reasoner`, `r1`, or `think` | Enables expert/reasoning mode |
| `search` or `online` | Enables upstream web search |
| Any other value | Standard chat mode |

On the Responses endpoint, `reasoning.effort` values of `medium` or `high`
also enable DeepSeek expert mode. Streaming reasoning fragments are emitted as
`response.reasoning_summary_text.delta` events.

For Qwen, a model name beginning with `qwen` is forwarded upstream. For
ChatGPT, the supplied model string is forwarded directly. Advertised model
lists are static compatibility entries defined in `app.py`; they are not
dynamically discovered.

## Client Examples

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-proxy-key",
    base_url="http://localhost:8000/ds/v1",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
```

### Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-proxy-key",
    base_url="http://localhost:8000/qwen",
)

response = client.messages.create(
    model="qwen3.7-plus",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.content[0].text)
```

Install `openai` or `anthropic` separately when using these examples; they are
not server dependencies.

### Codex Configuration

The included `codex_config.toml` demonstrates a custom provider:

```toml
model = "qwen3.7-plus"
model_provider = "qwen"

[model_providers.qwen]
name = "Qwen Custom Endpoint"
base_url = "http://localhost:8000/qwen/v1"
```

If proxy authentication is enabled, configure the client to send
`Authorization: Bearer <API_KEY>` using its supported environment-key or
custom-header setting.

## Current Compatibility Notes

This service implements the fields needed by its routes, but it is not a
complete implementation of every upstream public API.

- Token usage values are currently reported as zero.
- Sampling and token-limit fields are accepted by several request models but
  are not forwarded consistently to the web backends.
- `previous_response_id`, `session_id`, `chat_id`, and `conversation_id` are
  accepted for compatibility but do not select independent proxy sessions.
- One upstream chat/session is created per adapter at startup and reused for
  requests during that process.
- Message history after the latest assistant message is used to construct the
  next upstream prompt.
- Static model entries do not guarantee that the account can access a model.
- Internal provider APIs and anti-bot challenges may change at any time.

Because the current upstream session is shared, use this proxy as a
single-user/local-development service unless you add per-client isolation.

## Logging and Security

Request message content is written to `app.log`. This can include prompts,
tool results, source code, or other sensitive material. Protect, rotate, or
disable this log before using the service with confidential data.

For safer operation:

- Bind to `127.0.0.1` unless remote access is required.
- Set a strong `API_KEY` before exposing the server to a network.
- Put TLS and access controls in front of the service for remote use.
- Treat `.env`, cookies, bearer tokens, and `app.log` as secrets.
- Restart the server after rotating upstream credentials.

The health endpoints are not protected by `API_KEY`.

## Troubleshooting

### Adapter reports `unavailable`

Check that the adapter's cookie variable is present, restart the server, and
review startup errors. A configured adapter is marked unavailable when its
initial session cannot be created.

### `401 Invalid or missing API key`

Send `Authorization: Bearer <value>` using the exact value configured in
`API_KEY`, or leave `API_KEY` empty for local unauthenticated use.

### DeepSeek WASM file error

Run the application from the repository root and confirm that
`ds_api/sha3_wasm_bg.wasm` exists. If it is elsewhere, set `WASM_PATH`.

### Upstream authentication failure

Browser-session credentials expire. Sign in again, refresh the relevant
cookies or token in `.env`, and restart the proxy.

### Provider behavior changed

Failures after a provider website update may require changes in
`ds_api/adaptor.py`. Undocumented endpoints cannot offer the stability of
official APIs.

## Development

Basic syntax validation:

```bash
python -m py_compile app.py ds_api/adaptor.py ds_api/tool_dsml.py ds_api/tool_sieve.py
```

There is currently no automated test suite. When changing an adapter or
response translator, validate both streaming and non-streaming requests
against each configured provider.

## License

No license file is currently included. Unless a license is added, standard
copyright restrictions apply.