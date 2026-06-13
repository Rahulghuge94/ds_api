"""
DS Proxy — FastAPI Server
================================
Three API surface families:

  OpenAI-compatible    POST /v1/chat/completions
  Anthropic-compatible POST /v1/messages
  Codex-compatible     POST /v1/responses
                       POST /v1/completions
                       POST /v1/engines/{engine}/completions

All three call DSAdapter under the hood, translate request ↔ response
payloads, and pass DSML tool calls through StreamSieve / parse_dsml_tool_calls.

Environment variables (or .env file):
  DEEPSEEK_TOKEN   — Bearer token for chat.deepseek.com
  DEEPSEEK_COOKIES — Session cookies string
  API_KEY          — Secret key clients must pass as Bearer (leave empty to disable)
  WASM_PATH        — Path to sha3_wasm_bg.wasm  (default: ./sha3_wasm_bg.wasm)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ── Local modules ────────────────────────────────────────────────────────────
import sys, pathlib

_HERE = pathlib.Path(__file__).parent
for _p in (_HERE, _HERE / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv()

_API_KEY   = os.getenv("API_KEY", "")
_WASM_PATH = os.getenv("WASM_PATH", "ds_api/sha3_wasm_bg.wasm")

# Patch wasm bytes before importing adaptor (adaptor reads the file at import time)
import ds_api.adaptor as _adaptor_mod
with open(_WASM_PATH, "rb") as _wf:
    _adaptor_mod._WASM_BYTES = _wf.read()

from ds_api.adaptor import DeepSeekAdapter, QwenAdapter
from ds_api.tool_dsml import (                       
    parse_dsml_tool_calls,
    build_dsml_tool_prompt,
)
from ds_api.tool_sieve import StreamSieve            

ds_adapter: DeepSeekAdapter | None = None
qw_adapter: QwenAdapter | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ds_adapter, qw_adapter
    ds_adapter = DeepSeekAdapter()
    qw_adapter = QwenAdapter()
    yield
    if ds_adapter and ds_adapter._client:
        ds_adapter._client.close()
    if qs_adapter and ds_adapter._client:
        qs_adapter._client.close()
    
app = FastAPI(
    title="DS/QW Proxy",
    description=(
        "OpenAI / Anthropic / Codex compatible interface over DS/QW Chat "
        "with DSML tool-call support."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_bearer = HTTPBearer(auto_error=False)

def _check_auth(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    if not _API_KEY:
        return  # guard disabled
    if creds is None or creds.credentials != _API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key")

def _get_adapter(name: str) -> DeepSeekAdapter|QwenAdapter:
    _adapter = qw_adapter if name == "qwen" else ds_adapter
    
    if _adapter is None:
        raise HTTPException(500, "Adapter not initialised")
    return _adapter

def _model_to_ds(model: str) -> tuple[str | None, bool, bool]:
    """Map model string → (model_type, thinking_enabled, search_enabled)."""
    m = model.lower()
    if any(k in m for k in ("reasoner", "r1", "think")):
        return "expert", True, False
    if any(k in m for k in ("search", "online")):
        return None, False, True
    return None, False, False

def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten an OpenAI-style messages list into a single prompt string."""
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
        tag = {"system": "[System]", "assistant": "[Assistant]"}.get(role, "[User]")
        parts.append(f"{tag}: {content}")
    return "\n\n".join(parts)

def _responses_input_to_messages(inp: Any, ) -> list[dict]:
    """
    Normalise the `input` field of a Responses API request into a flat
    OpenAI-style messages list.

    Accepted forms:
      - plain string  → single user message
      - list of ResponsesInputItem dicts
      - list of {"type": "message", "role": ..., "content": ...} (SDK v2 format)
    """
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    if not isinstance(inp, list):
        return [{"role": "user", "content": str(inp)}]

    msgs = []
    for item in inp:
        if isinstance(item, str):
            msgs.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        # SDK v2 wraps items as {"type": "message", "role": ..., "content": ...}
        role    = item.get("role") or "user"
        content = item.get("content", "")
        # content can be a list of typed parts
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if p.get("type") in ("input_text", "text", "output_text")
            )
        msgs.append({"role": role, "content": content})
    return msgs

async def _run_sync(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

def _inject_tools(system: str | None, tools: list[dict] | None) -> str | None:
    if not tools:
        return system
    tool_prompt = build_dsml_tool_prompt(tools)
    return f"{system}\n\n{tool_prompt}" if system else tool_prompt

async def _prepare_session(
    messages: list[dict],
    system: str | None,
    tools: list[dict] | None,
    model: str,
) -> tuple[str, str, str | None, bool, bool]:
    """
    Build prompt + create DS session.
    Returns (session_id, prompt, model_type, thinking_enabled, search_enabled).
    """
    model_type, thinking_enabled, search_enabled = _model_to_ds(model)
    sys_prompt = _inject_tools(system, tools)
    if sys_prompt:
        messages = [{"role": "system", "content": sys_prompt}] + list(messages)
    prompt = _messages_to_prompt(messages)
    session_id = await _run_sync(_get_adapter().create_session)
    return session_id, prompt, model_type, thinking_enabled, search_enabled

def _rand_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"

def _sse(data: Any) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

_SSE_DONE = "data: [DONE]\n\n"

class OAIFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class OAITool(BaseModel):
    type: str = "function"
    function: OAIFunction


class OAIMessage(BaseModel):
    role: str
    content: Any = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class OAIChatRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: list[OAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[OAITool]] = None
    tool_choice: Optional[Any] = None
    system: Optional[str] = None   # convenience alias


# ─ Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicMessage(BaseModel):
    role: str
    content: Any  # str | list[{"type": "text", "text": "..."}]


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None


class AnthropicRequest(BaseModel):
    model: str = "deepseek-chat"
    max_tokens: int = 1024
    messages: list[AnthropicMessage]
    system: Optional[str] = None
    tools: Optional[list[AnthropicTool]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None


# ─ Codex / legacy ────────────────────────────────────────────────────────────

class CodexRequest(BaseModel):
    model: str = "code-davinci-002"
    prompt: Any = ""       # str | list[str]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = None
    stream: bool = False
    stop: Optional[Any] = None
    echo: bool = False
    suffix: Optional[str] = None


# ─ Responses API (OpenAI Responses / Codex CLI) ──────────────────────────────

class ResponsesInputItem(BaseModel):
    """One turn in the `input` array — mirrors the Responses API message format."""
    role: str                       # "user" | "assistant" | "system" | "developer"
    content: Any                    # str | list[{"type":"input_text","text":"..."}]


class ResponsesTool(BaseModel):
    type: str                       # "function" | "web_search" | "file_search" | "computer_use_preview"
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ResponsesRequest(BaseModel):
    model: str = "deepseek-chat"
    input: Any                      # str | list[ResponsesInputItem]
    instructions: Optional[str] = None   # system / developer prompt
    tools: Optional[list[ResponsesTool]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None   # stateful chaining (ignored here)
    tool_choice: Optional[Any] = None
    reasoning: Optional[dict] = None             # {"effort": "low"|"medium"|"high"}


# ══════════════════════════════════════════════════════════════════════════════
# Streaming generators
# ══════════════════════════════════════════════════════════════════════════════

async def _token_queue(
    adapter: DeepSeekAdapter,
    session_id: str,
    prompt: str,
    model_type: str | None,
    thinking_enabled: bool,
    search_enabled: bool,
) -> asyncio.Queue:
    """Spin a thread-based producer; return a filled queue."""
    q: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _produce():
        try:
            for tok in adapter.chat_stream(
                session_id, prompt,
                model_type=model_type,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            ):
                q.put_nowait(tok)
        except Exception as exc:
            q.put_nowait(exc)
        finally:
            q.put_nowait(None)

    await loop.run_in_executor(None, _produce)
    return q


async def _oai_stream_gen(
    session_id: str, prompt: str, model: str,
    model_type: str | None, thinking_enabled: bool, search_enabled: bool,
    chat_id: str, has_tools: bool,
) -> AsyncIterator[str]:
    adapter  = _get_adapter()
    sieve    = StreamSieve(parse_fn=parse_dsml_tool_calls if has_tools else None)
    created  = int(time.time())

    yield _sse({"id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    q = await _token_queue(adapter, session_id, prompt,
                           model_type, thinking_enabled, search_enabled)
    collected_tcs: list[dict] = []

    while True:
        tok = await q.get()
        if tok is None:
            break
        if isinstance(tok, Exception):
            raise tok
        if isinstance(tok, dict):
            continue  # skip thinking / status meta
        for ev in sieve.feed(tok):
            if ev.type == "text":
                yield _sse({"id": chat_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": ev.data}, "finish_reason": None}]})
            elif ev.type == "tool_calls":
                collected_tcs.extend(ev.data)

    for ev in sieve.flush():
        if ev.type == "text" and ev.data:
            yield _sse({"id": chat_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": ev.data}, "finish_reason": None}]})
        elif ev.type == "tool_calls":
            collected_tcs.extend(ev.data)

    finish = "stop"
    if collected_tcs:
        finish = "tool_calls"
        yield _sse({"id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"tool_calls": collected_tcs}, "finish_reason": None}]})

    yield _sse({"id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    yield _SSE_DONE


async def _anthropic_stream_gen(
    session_id: str, prompt: str, model: str,
    model_type: str | None, thinking_enabled: bool, search_enabled: bool,
    msg_id: str, has_tools: bool,
) -> AsyncIterator[str]:
    adapter = _get_adapter()
    sieve   = StreamSieve(parse_fn=parse_dsml_tool_calls if has_tools else None)

    yield _sse({"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})
    yield _sse({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
    yield _sse({"type": "ping"})

    q = await _token_queue(adapter, session_id, prompt,
                           model_type, thinking_enabled, search_enabled)
    collected_tcs: list[dict] = []

    while True:
        tok = await q.get()
        if tok is None:
            break
        if isinstance(tok, Exception):
            raise tok
        if isinstance(tok, dict):
            continue
        for ev in sieve.feed(tok):
            if ev.type == "text":
                yield _sse({"type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": ev.data}})
            elif ev.type == "tool_calls":
                collected_tcs.extend(ev.data)

    for ev in sieve.flush():
        if ev.type == "text" and ev.data:
            yield _sse({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": ev.data}})
        elif ev.type == "tool_calls":
            collected_tcs.extend(ev.data)

    yield _sse({"type": "content_block_stop", "index": 0})

    stop_reason = "end_turn"
    if collected_tcs:
        stop_reason = "tool_use"
        for idx, tc in enumerate(collected_tcs, start=1):
            fn = tc.get("function", {})
            try:
                inp = json.loads(fn.get("arguments", "{}"))
            except Exception:
                inp = {}
            yield _sse({"type": "content_block_start", "index": idx,
                        "content_block": {"type": "tool_use", "id": tc.get("id", ""),
                                          "name": fn.get("name", ""), "input": {}}})
            yield _sse({"type": "content_block_delta", "index": idx,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": json.dumps(inp, ensure_ascii=False)}})
            yield _sse({"type": "content_block_stop", "index": idx})

    yield _sse({"type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0}})
    yield _sse({"type": "message_stop"})
    yield _SSE_DONE


async def _codex_stream_gen(
    session_id: str, prompt: str, model: str,
    model_type: str | None, thinking_enabled: bool, search_enabled: bool,
    cmpl_id: str, created: int,
) -> AsyncIterator[str]:
    adapter = _get_adapter()
    q = await _token_queue(adapter, session_id, prompt,
                           model_type, thinking_enabled, search_enabled)
    while True:
        tok = await q.get()
        if tok is None:
            break
        if isinstance(tok, Exception):
            raise tok
        if isinstance(tok, dict):
            continue
        yield _sse({"id": cmpl_id, "object": "text_completion",
                    "created": created, "model": model,
                    "choices": [{"text": tok, "index": 0, "logprobs": None, "finish_reason": None}]})

    yield _sse({"id": cmpl_id, "object": "text_completion",
                "created": created, "model": model,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}]})
    yield _SSE_DONE


async def _responses_stream_gen(
    session_id: str, prompt: str, model: str,
    model_type: str | None, thinking_enabled: bool, search_enabled: bool,
    resp_id: str, has_tools: bool,
) -> AsyncIterator[str]:
    """
    Yield SSE in the OpenAI Responses API streaming format.

    Event sequence:
      response.created
      response.in_progress
      response.output_item.added        (output_text item)
      response.content_part.added
      response.output_text.delta  ×N
      [response.output_item.added  ← function_call item, if tool calls]
      response.output_item.done
      response.content_part.done
      response.completed
    """
    adapter = _get_adapter()
    sieve   = StreamSieve(parse_fn=parse_dsml_tool_calls if has_tools else None)
    created = int(time.time())

    base = {
        "id": resp_id, "object": "realtime.response", "model": model,
        "created_at": created, "status": "in_progress",
        "output": [], "usage": None,
    }

    def _ev(event_type: str, payload: dict) -> str:
        return _sse({"type": event_type, **payload})

    yield _ev("response.created",     {"response": {**base, "status": "in_progress"}})
    yield _ev("response.in_progress", {"response": base})

    # output item 0 — text
    yield _ev("response.output_item.added", {
        "response_id": resp_id, "output_index": 0,
        "item": {"id": _rand_id("item"), "type": "message", "role": "assistant",
                 "content": [], "status": "in_progress"},
    })
    yield _ev("response.content_part.added", {
        "response_id": resp_id, "item_index": 0, "output_index": 0,
        "part": {"type": "output_text", "text": ""},
    })

    q = await _token_queue(adapter, session_id, prompt,
                           model_type, thinking_enabled, search_enabled)
    collected_tcs: list[dict] = []
    full_text = ""

    while True:
        tok = await q.get()
        if tok is None:
            break
        if isinstance(tok, Exception):
            raise tok
        if isinstance(tok, dict):
            # Emit reasoning tokens as a separate event so Codex CLI can surface them
            if tok.get("__type") == "thinking" and tok.get("content"):
                yield _ev("response.reasoning_summary_text.delta",
                           {"delta": tok["content"], "response_id": resp_id})
            continue
        for ev in sieve.feed(tok):
            if ev.type == "text":
                full_text += ev.data
                yield _ev("response.output_text.delta", {
                    "response_id": resp_id, "item_index": 0, "output_index": 0,
                    "delta": ev.data,
                })
            elif ev.type == "tool_calls":
                collected_tcs.extend(ev.data)

    for ev in sieve.flush():
        if ev.type == "text" and ev.data:
            full_text += ev.data
            yield _ev("response.output_text.delta", {
                "response_id": resp_id, "item_index": 0, "output_index": 0,
                "delta": ev.data,
            })
        elif ev.type == "tool_calls":
            collected_tcs.extend(ev.data)

    yield _ev("response.output_text.done", {
        "response_id": resp_id, "item_index": 0, "output_index": 0, "text": full_text,
    })
    yield _ev("response.content_part.done", {
        "response_id": resp_id, "item_index": 0, "output_index": 0,
        "part": {"type": "output_text", "text": full_text},
    })

    # Emit function_call items for each tool call
    for tc_idx, tc in enumerate(collected_tcs, start=1):
        fn  = tc.get("function", {})
        try:
            args_str = fn.get("arguments", "{}")
        except Exception:
            args_str = "{}"
        call_id = tc.get("id", _rand_id("call"))
        yield _ev("response.output_item.added", {
            "response_id": resp_id, "output_index": tc_idx,
            "item": {
                "id": call_id, "type": "function_call",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": args_str,
                "status": "completed",
            },
        })
        yield _ev("response.output_item.done", {
            "response_id": resp_id, "output_index": tc_idx,
            "item": {
                "id": call_id, "type": "function_call",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": args_str,
                "status": "completed",
            },
        })

    status = "completed"
    yield _ev("response.output_item.done", {
        "response_id": resp_id, "output_index": 0,
        "item": {"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": full_text}]},
    })
    yield _ev("response.completed", {
        "response": {
            **base,
            "status": status,
            "output": [
                {"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": full_text}]},
                *[
                    {"type": "function_call", "call_id": tc.get("id", ""),
                     "name": tc.get("function", {}).get("name", ""),
                     "arguments": tc.get("function", {}).get("arguments", "{}")}
                    for tc in collected_tcs
                ],
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    })
    yield _SSE_DONE




@app.post("/v1/chat/completions", dependencies=[Depends(_check_auth)])
async def oai_chat_completions(req: OAIChatRequest):
    """OpenAI Chat Completions — streaming and non-streaming with tool calls."""
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools_raw = (
        [{"function": t.function.model_dump()} for t in req.tools] if req.tools else None
    )
    system = req.system or next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    non_sys = [m for m in messages if m["role"] != "system"]

    session_id, prompt, model_type, thinking, search = await _prepare_session(
        non_sys, system, tools_raw, req.model
    )
    chat_id = _rand_id("chatcmpl")

    if req.stream:
        return StreamingResponse(
            _oai_stream_gen(session_id, prompt, req.model, model_type,
                            thinking, search, chat_id, bool(tools_raw)),
            media_type="text/event-stream",
        )

    raw = await _run_sync(_get_adapter().chat, session_id, prompt,
                          model_type=model_type,
                          thinking_enabled=thinking,
                          search_enabled=search)
    tool_calls, cleaned = parse_dsml_tool_calls(raw)

    msg: dict[str, Any] = {"role": "assistant", "content": cleaned or None}
    finish = "stop"
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = None
        finish = "tool_calls"

    return JSONResponse({
        "id": chat_id, "object": "chat.completion",
        "created": int(time.time()), "model": req.model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# ══════════════════════════════════════════════════════════════════════════════
# Routes — Anthropic
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/messages", dependencies=[Depends(_check_auth)])
async def anthropic_messages(req: AnthropicRequest):
    """Anthropic Messages API — streaming and non-streaming with tool_use blocks."""
    messages = [m.model_dump() for m in req.messages]
    tools_raw: list[dict] | None = None
    if req.tools:
        tools_raw = [
            {"function": {"name": t.name, "description": t.description,
                          "parameters": t.input_schema}}
            for t in req.tools
        ]

    session_id, prompt, model_type, thinking, search = await _prepare_session(
        messages, req.system, tools_raw, req.model
    )
    msg_id = _rand_id("msg")

    if req.stream:
        return StreamingResponse(
            _anthropic_stream_gen(session_id, prompt, req.model, model_type,
                                  thinking, search, msg_id, bool(tools_raw)),
            media_type="text/event-stream",
        )

    raw = await _run_sync(_get_adapter().chat, session_id, prompt,
                          model_type=model_type,
                          thinking_enabled=thinking,
                          search_enabled=search)
    tool_calls, cleaned = parse_dsml_tool_calls(raw)

    content: list[dict] = []
    stop_reason = "end_turn"
    if cleaned:
        content.append({"type": "text", "text": cleaned})
    if tool_calls:
        stop_reason = "tool_use"
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                inp = json.loads(fn.get("arguments", "{}"))
            except Exception:
                inp = {}
            content.append({"type": "tool_use", "id": tc.get("id", ""),
                             "name": fn.get("name", ""), "input": inp})

    return JSONResponse({
        "id": msg_id, "type": "message", "role": "assistant", "model": req.model,
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    })


# ══════════════════════════════════════════════════════════════════════════════
# Routes — Codex / legacy completions
# ══════════════════════════════════════════════════════════════════════════════

async def _codex_handler(req: CodexRequest, engine: str | None = None) -> Any:
    """Shared handler for /v1/completions and /v1/engines/{engine}/completions."""
    model = engine or req.model
    model_type, thinking, search = _model_to_ds(model)

    prompt_text = "\n".join(req.prompt) if isinstance(req.prompt, list) else str(req.prompt)
    session_id  = await _run_sync(_get_adapter().create_session)
    cmpl_id     = _rand_id("cmpl")
    created     = int(time.time())

    if req.stream:
        return StreamingResponse(
            _codex_stream_gen(session_id, prompt_text, model,
                              model_type, thinking, search, cmpl_id, created),
            media_type="text/event-stream",
        )

    text = await _run_sync(_get_adapter().chat, session_id, prompt_text,
                           model_type=model_type, thinking_enabled=thinking,
                           search_enabled=search)
    if req.echo:
        text = prompt_text + text
    if req.suffix:
        text += req.suffix

    return JSONResponse({
        "id": cmpl_id, "object": "text_completion",
        "created": created, "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.post("/v1/completions", dependencies=[Depends(_check_auth)])
async def completions(req: CodexRequest):
    """Legacy /v1/completions endpoint (OpenAI text completion format)."""
    return await _codex_handler(req)


@app.post("/v1/engines/{engine}/completions", dependencies=[Depends(_check_auth)])
async def codex_completions(engine: str, req: CodexRequest):
    """Classic Codex endpoint: /v1/engines/{engine}/completions."""
    return await _codex_handler(req, engine=engine)

def _reasoning_to_ds(reasoning: dict | None) -> tuple[bool, str | None]:
    """Map reasoning.effort → (thinking_enabled, model_type override)."""
    if not reasoning:
        return False, None
    effort = (reasoning.get("effort") or "").lower()
    if effort in ("medium", "high"):
        return True, "expert"
    return False, None

@app.post("/v1/responses", dependencies=[Depends(_check_auth)])
async def responses(req: ResponsesRequest):
    """
    OpenAI Responses API — used by Codex CLI (openai-cli) and openai-python SDK v2+.

    Supports:
      • plain text and multi-part input arrays
      • instructions  (system/developer prompt)
      • function tools  (translated via DSML)
      • reasoning.effort → DeepSeek expert/thinking mode
      • streaming (response.* SSE events) and non-streaming
    """
    messages = _responses_input_to_messages(req.input)

    # Build tools_raw from Responses-API tool format
    tools_raw: list[dict] | None = None
    if req.tools:
        tools_raw = []
        for t in req.tools:
            if t.type == "function" and t.name:
                tools_raw.append({"function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.parameters or {},
                }})
            # web_search / file_search / computer_use_preview — no-op shims
            # (DS has its own search; we just skip unknown tool types)

    # reasoning.effort can force thinking mode independently of model name
    thinking_override, model_type_override = _reasoning_to_ds(req.reasoning)
    model_type, thinking, search = _model_to_ds(req.model)
    if thinking_override:
        thinking   = True
        model_type = model_type_override or model_type

    sys_prompt = _inject_tools(req.instructions, tools_raw)
    if sys_prompt:
        messages = [{"role": "system", "content": sys_prompt}] + messages

    prompt     = _messages_to_prompt(messages)
    session_id = await _run_sync(_get_adapter().create_session)
    resp_id    = _rand_id("resp")
    created    = int(time.time())

    if req.stream:
        return StreamingResponse(
            _responses_stream_gen(session_id, prompt, req.model, model_type,
                                  thinking, search, resp_id, bool(tools_raw)),
            media_type="text/event-stream",
        )

    # ── Non-streaming ──────────────────────────────────────────────────────
    raw = await _run_sync(_get_adapter().chat, session_id, prompt,
                          model_type=model_type,
                          thinking_enabled=thinking,
                          search_enabled=search)
    tool_calls, cleaned = parse_dsml_tool_calls(raw)

    output: list[dict] = []
    if cleaned:
        output.append({
            "type": "message", "id": _rand_id("msg"),
            "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": cleaned}],
        })
    for tc in tool_calls:
        fn = tc.get("function", {})
        output.append({
            "type": "function_call",
            "id": tc.get("id", _rand_id("call")),
            "call_id": tc.get("id", _rand_id("call")),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
            "status": "completed",
        })

    return JSONResponse({
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "model": req.model,
        "status": "completed",
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": req.previous_response_id,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    })




_MODELS = [
    {"id": "deepseek-chat",     "object": "model", "created": 0, "owned_by": "deepseek"},
    {"id": "deepseek-reasoner", "object": "model", "created": 0, "owned_by": "deepseek"},
    {"id": "deepseek-r1",       "object": "model", "created": 0, "owned_by": "deepseek"},
    {"id": "deepseek-search",   "object": "model", "created": 0, "owned_by": "deepseek"},
    # Codex CLI defaults — shim so model-listing calls don't 404
    {"id": "codex-mini-latest", "object": "model", "created": 0, "owned_by": "openai"},
    {"id": "o4-mini",           "object": "model", "created": 0, "owned_by": "openai"},
    {"id": "o3",                "object": "model", "created": 0, "owned_by": "openai"},
    {"id": "gpt-4o",            "object": "model", "created": 0, "owned_by": "openai"},
    # Legacy completions
    {"id": "code-davinci-002",  "object": "model", "created": 0, "owned_by": "openai"},
    {"id": "code-cushman-001",  "object": "model", "created": 0, "owned_by": "openai"},
]


@app.get("/v1/models", dependencies=[Depends(_check_auth)])
async def list_models():
    return JSONResponse({"object": "list", "data": _MODELS})

@app.get("/v1/models/{model_id}", dependencies=[Depends(_check_auth)])
async def get_model(model_id: str):
    for m in _MODELS:
        if m["id"] == model_id:
            return JSONResponse(m)
    raise HTTPException(404, f"Model '{model_id}' not found")

@app.get("/health")
async def health():
    return {"status": "ok", "adapter_ready": _adapter is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
