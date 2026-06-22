"""
DeepSeek Chat API Adapter - WASM-based PoW solving, session management, streaming
Supports expert mode (thinking_enabled, search_enabled).
"""
from __future__ import annotations

import json
import os
import time
import struct
import uuid
import base64
import datetime
import hashlib
import random
import re
import threading
import httpx
from dotenv import load_dotenv
from wasmtime import Store, Module, Instance
from base64 import b64encode

load_dotenv()
# deepseek conf
COOKIES = os.environ.get("DEEPSEEK_COOKIES", "")
BASE_URL = "https://chat.deepseek.com"
TOKEN = os.environ.get("DEEPSEEK_TOKEN", "")
# qwen conf
QWEN_TOKEN = os.environ.get("QWEN_TOKEN", "")
QWEN_COOKIES = os.environ.get("QWEN_COOKIES", "")
QWEN_BASE_URL = "https://chat.qwen.ai"
DEFAULT_QWEN_MODEL = "qwen3.7-plus"
# chatgpt conf
CHATGPT_BASE_URL = "https://chatgpt.com"
CHATGPT_TOKEN = os.environ.get("CHATGPT_TOKEN", "")
CHATGPT_COOKIES = (
    os.environ.get("CHATGPT_COOKIES", "")
    or os.environ.get("CHATGPT_COOKIE_PART_1", "")
    + os.environ.get("CHATGPT_COOKIE_PART_2", "")
)
DEFAULT_CHATGPT_MODEL = "auto"

with open("ds_api/sha3_wasm_bg.wasm", "rb") as f:
    _WASM_BYTES = f.read()

class WASMError(Exception):
    pass

class PoWError(Exception):
    pass

class _WASMSolver:
    """WASM-based PoW solver (reused across calls) — thread-safe via lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self.store = Store()
        module = Module(self.store.engine, _WASM_BYTES)
        instance = Instance(self.store, module, [])
        exports = instance.exports(self.store)
        self.memory = exports["memory"]
        self.wasm_solve = exports["wasm_solve"]
        self.add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        self.malloc = exports["__wbindgen_export_0"]
        self._wbindgen_free = exports["__wbindgen_export_2"]
        self._allocations: list[tuple[int, int]] = []

    def _encode(self, s: str):
        data = s.encode("utf-8")
        ptr = self.malloc(self.store, len(data), 1)
        mem = self.memory.data_ptr(self.store)
        for i, b in enumerate(data):
            mem[ptr + i] = b
        self._allocations.append((ptr, len(data)))
        return ptr, len(data)

    def _free_allocations(self):
        for ptr, length in self._allocations:
            try:
                self._wbindgen_free(self.store, ptr, length, 1)
            except Exception:
                pass
        self._allocations.clear()

    def solve(self, challenge: str, salt: str, expire_at: int, difficulty: int) -> int:
        with self._lock:
            try:
                prefix = f"{salt}_{expire_at}_"
                stack_ptr = self.add_to_stack(self.store, -16)
                chal_ptr, chal_len = self._encode(challenge)
                prefix_ptr, prefix_len = self._encode(prefix)
                self.wasm_solve(self.store, stack_ptr, chal_ptr, chal_len,
                                prefix_ptr, prefix_len, float(difficulty))
                mem = self.memory.data_ptr(self.store)
                ret = int.from_bytes(bytes(mem[stack_ptr:stack_ptr + 4]),
                                     byteorder='little', signed=True)
                if ret == 0:
                    raise PoWError("WASM solver found no solution")
                result = struct.unpack('<d', bytes(mem[stack_ptr + 8:stack_ptr + 16]))[0]
                self.add_to_stack(self.store, 16)
                return int(result)
            finally:
                self._free_allocations()

class DeepSeekAdapter:
    """Adapter for DeepSeek Chat API"""

    def __init__(self, token: str = TOKEN, cookies: str = COOKIES):
        self.token = token
        self.cookies = cookies
        self._solver = None
        self._client = httpx.Client(timeout=120)
        self._msg_counters: dict[str, int] = {}
        self._base_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Cookie": cookies,
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/",
            "X-App-Version": "20241129.1",
            "X-Client-Version": "2.0.0",
            "X-Client-Platform": "web",
            "X-Client-Locale": "zh_CN",
            "X-Client-Timezone-Offset": "28800",
        }

    @property
    def solver(self):
        if self._solver is None:
            self._solver = _WASMSolver()
        return self._solver

    def _get_challenge(self, target_path: str = "/api/v0/chat/completion"):
        resp = self._client.post(
            f"{BASE_URL}/api/v0/chat/create_pow_challenge",
            json={"target_path": target_path},
            headers=self._base_headers,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["data"]["biz_data"]["challenge"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected challenge response structure: {data.get('code', 'unknown')} - {data.get('msg', str(e))}"
            )

    def _solve(self, challenge_data: dict) -> str:
        nonce = self.solver.solve(
            challenge=challenge_data["challenge"],
            salt=challenge_data["salt"],
            expire_at=challenge_data["expire_at"],
            difficulty=challenge_data["difficulty"],
        )
        raw = json.dumps({
            "algorithm": "DeepSeekHashV1",
            "challenge": challenge_data["challenge"],
            "salt": challenge_data["salt"],
            "answer": nonce,
            "signature": challenge_data["signature"],
            "target_path": challenge_data["target_path"],
        }, separators=(",", ":"))
        return base64.b64encode(raw.encode()).decode()

    def _pow_headers(self, target_path: str = "/api/v0/chat/completion"):
        c = self._get_challenge(target_path)
        pow_h = self._solve(c)
        return {**self._base_headers, "X-DS-PoW-Response": pow_h}

    def create_session(self) -> str:
        """Create a new chat session, returns session_id"""
        headers = self._pow_headers("/api/v0/chat/completion")
        resp = self._client.post(
            f"{BASE_URL}/api/v0/chat_session/create",
            json={"target_path": "/api/v0/chat/completion"},
            headers=headers,
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Session creation failed: {data}")
        biz = data["data"]["biz_data"]
        # Handle both formats: direct id vs nested chat_session.id
        if "id" in biz:
            return biz["id"]
        return biz["chat_session"]["id"]

    def _parse_sse(self, text: str):
        """Parse SSE text into a list of events"""
        events = []
        current_event = ""
        for line in text.split("\n"):
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str:
                    try:
                        events.append((current_event, json.loads(data_str)))
                    except json.JSONDecodeError:
                        events.append((current_event, data_str))
                current_event = ""
        return events

    def _send_completion(self, session_id: str, prompt: str, stream: bool = False,
                         model_type: str | None = None,
                         thinking_enabled: bool = False, search_enabled: bool = False):
        """Send a completion request, returns raw response"""
        headers = self._pow_headers("/api/v0/chat/completion")
        # Auto-increment parent_message_id per session
        mid = self._msg_counters.get(session_id, 0) + 1
        self._msg_counters[session_id] = mid
        body = {
            "chat_session_id": session_id,
            "parent_message_id": mid - 1 if mid > 1 else None,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": [],
            "stream": stream,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }
        return self._client.post(
            f"{BASE_URL}/api/v0/chat/completion",
            json=body,
            headers=headers,
        )

    def chat(self, session_id: str, prompt: str, model_type: str | None = None,
             thinking_enabled: bool = False, search_enabled: bool = False) -> str:
        """Send a non-streaming chat message, returns response content."""
        resp = self._send_completion(session_id, prompt, stream=False,
                                      model_type=model_type,
                                      thinking_enabled=thinking_enabled,
                                      search_enabled=search_enabled)
        events = self._parse_sse(resp.text)

        # Collect all content from both normal mode and expert fragment mode
        content_parts = []
        thinking_parts = []
        frag_type = None  # None, 'thinking', 'content'

        for event_type, data in events:
            if not isinstance(data, dict):
                continue
            p = data.get("p", "")
            o = data.get("o", "")
            v = data.get("v", "")

            # Expert mode: initial response with fragments
            if isinstance(v, dict) and 'response' in v:
                resp_data = v['response']
                fragments = resp_data.get('fragments', [])
                if fragments:
                    ft = fragments[0].get('type', '')
                    frag_type = 'thinking' if ft == 'THINK' else 'content'
                    fc = fragments[0].get('content', '')
                    if fc:
                        (thinking_parts if frag_type == 'thinking' else content_parts).append(fc)
                continue

            # Expert mode: fragment content append
            if p == "response/fragments/-1/content" and o == "APPEND":
                if frag_type == 'thinking':
                    thinking_parts.append(v)
                else:
                    content_parts.append(v)
                continue
            if p == "response/fragments/-1/content" and not o:
                # Frag content without o (happens after fragment switch)
                if frag_type == 'thinking':
                    thinking_parts.append(v)
                else:
                    content_parts.append(v)
                continue

            # Expert mode: fragment switch
            if p == "response/fragments" and o == "APPEND":
                if isinstance(v, list) and v:
                    new_type = v[0].get('type', '')
                    if new_type == 'RESPONSE':
                        frag_type = 'content'
                    elif new_type == 'THINK':
                        frag_type = 'thinking'
                continue

            # Normal mode
            if p == "response/content" and o == "APPEND":
                content_parts.append(v)
                continue

            # Plain token event — belongs to current fragment or normal mode
            if "v" in data and "p" not in data and "o" not in data:
                token = data["v"]
                if isinstance(token, str) and token:
                    if frag_type == 'thinking':
                        thinking_parts.append(token)
                    else:
                        content_parts.append(token)
                continue

        return "".join(content_parts)

    def chat_stream(self, session_id: str, prompt: str,
                    model_type: str | None = None,
                    thinking_enabled: bool = False, search_enabled: bool = False):
        """Stream a chat message, yields content tokens.

        In expert mode (model_type='expert'), yields dicts with
        __type='thinking' for reasoning tokens and strings for final content.
        """
        headers = self._pow_headers("/api/v0/chat/completion")
        mid = self._msg_counters.get(session_id, 0) + 1
        self._msg_counters[session_id] = mid
        body = {
            "chat_session_id": session_id,
            "parent_message_id": mid - 1 if mid > 1 else None,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": [],
            "stream": True,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }
        with self._client.stream(
            "POST", f"{BASE_URL}/api/v0/chat/completion",
            json=body, headers=headers,
        ) as resp:
            frag_type = None  # None, 'thinking', 'content'

            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue

                p = data.get("p", "")
                o = data.get("o", "")
                v = data.get("v", "")

                # Initial response with fragments (expert mode)
                if isinstance(v, dict) and 'response' in v:
                    resp_data = v['response']
                    fragments = resp_data.get('fragments', [])
                    if fragments:
                        ft = fragments[0].get('type', '')
                        frag_type = 'thinking' if ft == 'THINK' else 'content'
                        fc = fragments[0].get('content', '')
                        if fc:
                            if frag_type == 'thinking':
                                yield {"__type": "thinking", "content": fc}
                            else:
                                yield fc
                    else:
                        frag_type = 'content'
                        content = resp_data.get('content', '')
                        if content:
                            yield content
                    continue

                # Fragment content append (expert mode)
                if p == "response/fragments/-1/content" and o == "APPEND":
                    if frag_type == 'thinking':
                        if v:
                            yield {"__type": "thinking", "content": v}
                    else:
                        if v:
                            yield v
                    continue

                # Fragment content without o (after fragment switch in batched responses)
                if p == "response/fragments/-1/content" and not o:
                    if frag_type == 'thinking':
                        if v:
                            yield {"__type": "thinking", "content": v}
                    else:
                        if v:
                            yield v
                    continue

                # Fragment switch (expert mode)
                if p == "response/fragments" and o == "APPEND":
                    if isinstance(v, list) and v:
                        new_type = v[0].get('type', '')
                        if new_type == 'RESPONSE':
                            frag_type = 'content'
                        elif new_type == 'THINK':
                            frag_type = 'thinking'
                    continue

                # Normal mode content
                if p == "response/content" and o == "APPEND":
                    yield v
                    continue

                # Plain token event
                if "v" in data and "p" not in data and "o" not in data:
                    token = data["v"]
                    if isinstance(token, str) and token:
                        if frag_type == 'thinking':
                            yield {"__type": "thinking", "content": token}
                        else:
                            yield token
                    continue

                # Status
                if p == "response/status":
                    yield {"__type": "status", "status": v}
                    continue

class QwenAdapter:
    """Adapter for chat.qwen.ai's internal chat completion API."""
 
    def __init__(self, token: str = QWEN_TOKEN, cookies: str = QWEN_COOKIES):
        self.token = token
        self.cookies = cookies
        self._client = httpx.Client(timeout=120)
        # Track the parent (last assistant) message id per chat session, so
        # multi-turn conversations link correctly.
        self._parent_ids: dict[str, str | None] = {}
        self._chat_types = ["t2t", "search", "t2i", "t2v", "deep_research"]

    def _headers(self) -> dict:
        h = {
            "X-Request-Id": str(uuid.uuid4()),
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": f"{QWEN_BASE_URL}/",
            "Origin": QWEN_BASE_URL,
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        }

        if self.token:
            h["Authorization"] = f"Bearer {self.token}"

        if self.cookies:
            h["cookies"] = self.cookies

        return h
 
    # ── Session management ───────────────────────────────────────────────
    def create_session(self, model: str = DEFAULT_QWEN_MODEL) -> str:
        """Create a new chat session on chat.qwen.ai, returns chat_id."""
        resp = self._client.post(
            f"{QWEN_BASE_URL}/api/v2/chats/new",
            json={"title": "New Chat", "models": [model], "chat_mode": "normal",
                  "chat_type": "t2t", "timestamp": int(time.time() * 1000)},
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        chat_id = (
            data.get("data", {}).get("id")
            or data.get("data", {}).get("chat", {}).get("id")
            or data.get("id")
        )
        if not chat_id:
            raise RuntimeError(f"Qwen session creation failed: {data}")
        self._parent_ids[chat_id] = None
        return chat_id
 
    # ── Request body construction ────────────────────────────────────────
 
    def _build_body(
        self,
        chat_id: str,
        prompt: str,
        model: str,
        thinking_enabled: bool,
        search_enabled: bool,
        stream: bool,
        thinking_budget: int = 38912,
        generate_image: bool=False,
        generate_video: bool=False,
        deep_research: bool=False,
    ) -> dict:
        msg_id = uuid.uuid4().hex
        parent_id = self._parent_ids.get(chat_id)
 
        feature_config: dict = {
            "thinking_enabled": thinking_enabled,
            "output_schema": "phase",
            "research_mode": "normal",
            "auto_thinking": True,
            "thinking_mode": "Auto",
            "thinking_format": "summary",
            "auto_search": True,
        }

        # check if
        chat_type = "t2t"
        meta = {"subChatType": "t2t"}
        if not generate_image and not generate_video and not deep_research and not search_enabled:
            chat_type = "t2t"
            
        elif generate_image and not generate_video and not deep_research and not search_enabled:
            chat_type = "t2i"
            meta = {"subChatType": "t2i", "size": "16:9"}
        elif generate_video and not generate_image and not deep_research and not search_enabled:
            chat_type = "t2v"
            meta = {"subChatType": "t2t", "size": "16:9"}
        elif deep_research and not generate_image and not generate_video and not search_enabled:
            chat_type = "deep_research"
            meta = {"subChatType": "deep_research"}
        elif search_enabled and not generate_image and not generate_video and not deep_research:
            chat_type = "search"
            meta = {"subChatType": "search"}

        message = {
            "fid": msg_id,
            "parentId": parent_id,
            "childrenIds": [str(uuid.uuid4())],
            "role": "user",
            "content": prompt,
            "user_action": "chat",
            "files": [],
            "timestamp": int(time.time()),
            "models": [model],
            "chat_type": chat_type,
            "feature_config": feature_config,
            "extra": {"meta": meta},
            "sub_chat_type": chat_type,
        }
 
        return {
            "stream": stream,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": parent_id,
            "messages": [message],
            "timestamp": int(time.time()),
        }
 
    def _send_completion(
        self,
        chat_id: str,
        prompt: str,
        stream: bool,
        model_type: str | None,
        thinking_enabled: bool,
        search_enabled: bool,
    ):
        model = model_type or DEFAULT_QWEN_MODEL
        body = self._build_body(chat_id, prompt, model, thinking_enabled,
                                 search_enabled, stream)

        url = f"{QWEN_BASE_URL}/api/v2/chat/completions?chat_id={chat_id}"
        if not stream:
            return self._client.post(url, json=body, headers=self._headers())
        return self._client.stream("POST", url, json=body, headers=self._headers())
 
    # ── Non-streaming chat ───────────────────────────────────────────────
 
    def chat(self, session_id: str, prompt: str, model_type: str | None = None,
             thinking_enabled: bool = False, search_enabled: bool = False) -> str:
        """Send a non-streaming chat message, returns response content."""
        content_parts = []
        for token in self.chat_stream(
            session_id,
            prompt,
            model_type=model_type,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
        ):
            if isinstance(token, str):
                content_parts.append(token)
        return "".join(content_parts)
 
    # ── Streaming chat ───────────────────────────────────────────────────
 
    def chat_stream(self, session_id: str, prompt: str,
                    model_type: str | None = None,
                    thinking_enabled: bool = False, search_enabled: bool = False):
        """Stream a chat message, yields content tokens.
 
        Mirrors DeepSeekAdapter.chat_stream: yields plain strings for content
        tokens, and dicts with __type='thinking'/'status' for metadata.
        """
        ctx = self._send_completion(session_id, prompt, stream=True,
                                     model_type=model_type,
                                     thinking_enabled=thinking_enabled,
                                     search_enabled=search_enabled)
        new_msg_id = None
        with ctx as resp:
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                resp.read()
                data = resp.json()
                if data.get("success") is False:
                    raise RuntimeError(f"Qwen completion failed: {data}")
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
 
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
 
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
 
                if not new_msg_id:
                    new_msg_id = data.get("response.created", {}).get("id") if isinstance(
                        data.get("response.created"), dict) else data.get("id")
 
                choices = data.get("choices") or []
                for choice in choices:
                    delta = choice.get("delta", {})
                    phase = delta.get("phase")
                    content = delta.get("content")
 
                    if content is None:
                        continue
 
                    if phase == "think":
                        if content:
                            yield {"__type": "thinking", "content": content}
                        continue
 
                    if phase == "answer" or phase is None:
                        if content:
                            yield content
                        continue
 
                    # Other phases (e.g. "tool_call") surfaced as status events
                    if content:
                        yield {"__type": "status", "status": phase}
 
        if new_msg_id:
            self._parent_ids[session_id] = new_msg_id

class ChatGPTAdapter:
    """Adapter for chatgpt.com's internal conversation API."""

    def __init__(
        self,
        token: str | None = CHATGPT_TOKEN,
        cookies: str = CHATGPT_COOKIES,
    ):
        self.token = token or None
        self.cookies = cookies
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0'
        self.session_max_messages = max(1, 50)
        self.session_ttl_seconds = max(1, 3 * 60 * 60)
        self._client = httpx.Client(timeout=120)
        self._sessions: dict[str, dict] = {}
        self._token_expiry = 0.0
        self._scripts_cache: dict | None = None
        self._lock = threading.RLock()

    def _headers(self, *, authorized: bool = False) -> dict[str, str]:
        headers = {
            "accept": "*/*",
            "accept-encoding": "application/json;",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "cookie": self.cookies,
            "oai-client-build-number": "6844271",
            "oai-client-version": "prod-8117aba90ffac2b43c6118f2fc2eefaac58f816d",
            "origin": CHATGPT_BASE_URL,
            "referer": f"{CHATGPT_BASE_URL}/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": self.user_agent,
        }

        if authorized:
            headers["Authorization"] = f"Bearer {self._get_token()}"
        return headers

    def _get_token(self) -> str:
        with self._lock:
            if self.token and (
                self._token_expiry == 0 or time.time() < self._token_expiry
            ):
                return self.token
            if not self.cookies:
                raise RuntimeError("Set CHATGPT_COOKIES")

            response = self._client.get(
                f"{CHATGPT_BASE_URL}/api/auth/session",
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            access_token = data.get("accessToken")
            if not access_token:
                raise RuntimeError("Not logged in. Check your ChatGPT cookies.")
            self.token = access_token
            self._token_expiry = time.time() + 300
            return access_token

    def _get_scripts_and_dpl(self) -> dict:
        with self._lock:
            if self._scripts_cache is not None:
                return self._scripts_cache
            try:
                response = self._client.get(
                    f"{CHATGPT_BASE_URL}/",
                    headers=self._headers(),
                )
                response.raise_for_status()
                scripts = re.findall(r'src="([^"]*)"', response.text)
                match = re.search(r"dpl=([a-zA-Z0-9_-]+)", response.text)
                self._scripts_cache = {
                    "scripts": scripts or [None],
                    "dpl": match.group(1) if match else "",
                }
            except httpx.HTTPError:
                self._scripts_cache = {"scripts": [None], "dpl": ""}
            return self._scripts_cache

    def _solve_pow(
        self,
        seed: str,
        difficulty: str,
        scripts: list[str | None] | None,
        dpl: str,
    ) -> str | None:
        start_time = time.time()
        navigator = {
            "appCodeName": "Mozilla",
            "appName": "Netscape",
            "appVersion": "5.0 (Windows NT 10.0; Win64; x64)",
            "cookieEnabled": True,
            "language": "en-US",
            "languages": ["en-US", "en"],
            "platform": "Win32",
            "userAgent": self.user_agent,
            "vendor": "Google Inc.",
            "hardwareConcurrency": 8,
        }
        navigator_key = random.choice(list(navigator))
        config = [
            navigator["hardwareConcurrency"] + 1920 + 1080,
            datetime.datetime.now().ctime(),
            4294705152,
            0,
            self.user_agent,
            random.choice(scripts or [None]),
            dpl or "",
            navigator["language"],
            ",".join(navigator["languages"]),
            0,
            f"{navigator_key}-{navigator[navigator_key]}",
            random.choice(["location", "cookie", "title", "createElement"]),
            random.choice(["fetch", "setTimeout", "console", "document"]),
            int(time.time() * 1000),
            str(uuid.uuid4()),
        ]

        for nonce in range(1, 100000):
            config[3] = nonce
            config[9] = round((time.time() - start_time) * 1000)
            encoded = base64.b64encode(
                json.dumps(config).encode("utf-8")
            ).decode("ascii")
            digest = hashlib.sha3_512((seed + encoded).encode()).hexdigest()
            if digest[:len(difficulty)] <= difficulty:
                return encoded
        return None

    def _get_requirements_and_pow(self) -> dict[str, str]:
        headers = self._headers(authorized=True)
        headers["Content-Type"] = "application/json"
        response = self._client.post(
            f"{CHATGPT_BASE_URL}/backend-api/sentinel/chat-requirements",
            headers=headers,
            json={"conversation_mode_kind": "primary_assistant"},
        )
        response.raise_for_status()
        requirements = response.json()
        result = {}
        if requirements.get("token"):
            result["requirementsToken"] = requirements["token"]

        proof = requirements.get("proofofwork") or {}
        if proof.get("required"):
            script_data = self._get_scripts_and_dpl()
            proof_token = self._solve_pow(
                proof["seed"],
                proof["difficulty"],
                script_data.get("scripts"),
                script_data.get("dpl", ""),
            )
            if not proof_token:
                raise RuntimeError("Unable to solve ChatGPT proof-of-work challenge.")
            result["proofToken"] = f"gAAAAAB{proof_token}"
        return result

    @staticmethod
    def _new_session_state() -> dict:
        now = time.time()
        return {
            "conversation_id": None,
            "parent_message_id": None,
            "message_count": 0,
            "created_at": now,
            "last_used_at": now,
        }

    def create_session(self, model: str = DEFAULT_CHATGPT_MODEL) -> str:
        """Create a local session; ChatGPT creates its conversation on first use."""
        del model
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = self._new_session_state()
        return session_id

    def _session_for_request(self, session_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(session_id)
            now = time.time()
            if (
                not session
                or session["message_count"] >= self.session_max_messages
                or now - session["created_at"] >= self.session_ttl_seconds
            ):
                session = self._new_session_state()
                self._sessions[session_id] = session
            return session.copy()

    @staticmethod
    def _iter_sse_data(response: httpx.Response):
        data_lines = []
        for raw_line in response.iter_lines():
            line = raw_line.rstrip("\r")
            if not line:
                if data_lines:
                    yield "".join(data_lines)
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            yield "".join(data_lines)

    def _conversation_headers(self) -> dict[str, str]:
        proof_data = self._get_requirements_and_pow()
        headers = self._headers(authorized=True)
        headers.update({
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "OAI-Language": "en-US",
        })
        device_match = re.search(r"(?:^|;\s*)oai-did=([^;]+)", self.cookies)
        if device_match:
            headers["OAI-Device-Id"] = device_match.group(1)
        if proof_data.get("requirementsToken"):
            headers["Openai-Sentinel-Chat-Requirements-Token"] = (
                proof_data["requirementsToken"]
            )
        if proof_data.get("proofToken"):
            headers["Openai-Sentinel-Proof-Token"] = proof_data["proofToken"]
        return headers

    @staticmethod
    def _build_body(session: dict, prompt: str, model: str) -> dict:
        offset = datetime.datetime.now().astimezone().utcoffset()
        payload = {
            "action": "next",
            "messages": [{
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
                "metadata": {},
            }],
            "model": model,
            "parent_message_id": (
                session.get("parent_message_id") or str(uuid.uuid4())
            ),
            "timezone_offset_min": int(offset.total_seconds() // 60) if offset else 0,
            "history_and_training_disabled": False,
            "conversation_mode": {"kind": "primary_assistant"},
            "force_paragen": False,
            "force_nulligen": False,
            "force_rate_limit": False,
        }
        if session.get("conversation_id"):
            payload["conversation_id"] = session["conversation_id"]
        return payload

    def chat(
        self,
        session_id: str,
        prompt: str,
        model_type: str | None = None,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
    ) -> str:
        """Send a message and collect the streamed response."""
        return "".join(
            token for token in self.chat_stream(
                session_id,
                prompt,
                model_type=model_type,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            )
            if isinstance(token, str)
        )

    def chat_stream(
        self,
        session_id: str,
        prompt: str,
        model_type: str | None = None,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
    ):
        """Stream text deltas from ChatGPT's cumulative SSE messages."""
        del thinking_enabled, search_enabled
        if not self.cookies and not self.token:
            raise RuntimeError(
                "Set CHATGPT_COOKIES or CHATGPT_COOKIE_PART_1/"
                "CHATGPT_COOKIE_PART_2."
            )

        session = self._session_for_request(session_id)
        payload = self._build_body(
            session, prompt, model_type or DEFAULT_CHATGPT_MODEL
        )
        headers = self._conversation_headers()
        full_content = ""
        conversation_id = session.get("conversation_id")
        assistant_message_id = None

        with self._client.stream(
            "POST",
            f"{CHATGPT_BASE_URL}/backend-api/conversation",
            headers=headers,
            json=payload,
            timeout=None,
        ) as response:
            if response.status_code >= 400:
                response.read()
                if (
                    response.status_code == 404
                    and "conversation_deleted" in response.text
                ):
                    with self._lock:
                        self._sessions[session_id] = self._new_session_state()
                response.raise_for_status()

            if "application/json" in response.headers.get("content-type", ""):
                response.read()
                raise RuntimeError(
                    f"Unexpected ChatGPT JSON response: {response.text}"
                )

            for event_data in self._iter_sse_data(response):
                if event_data == "[DONE]":
                    break
                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError:
                    continue

                conversation_id = event.get("conversation_id") or conversation_id
                message = event.get("message") or {}
                assistant_message_id = message.get("id") or assistant_message_id
                if (message.get("author") or {}).get("role") != "assistant":
                    continue
                parts = (message.get("content") or {}).get("parts")
                if not isinstance(parts, list):
                    continue

                new_content = "".join(
                    part for part in parts if isinstance(part, str)
                )
                delta = (
                    new_content[len(full_content):]
                    if new_content.startswith(full_content)
                    else new_content
                )
                full_content = new_content
                if delta:
                    yield delta

        if conversation_id:
            with self._lock:
                current = self._sessions.setdefault(
                    session_id, self._new_session_state()
                )
                current["conversation_id"] = conversation_id
                if assistant_message_id:
                    current["parent_message_id"] = assistant_message_id
                current["message_count"] += 1
                current["last_used_at"] = time.time()