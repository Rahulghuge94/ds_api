"""
StreamSieve
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class SieveEvent:
    type: str  # 'text' | 'tool_calls'
    data: Any


class StreamSieve:
    """实时筛分 SSE 流中的 DSML 工具调用。"""

    _TOOL_STARTS = [
        "<|DSML|tool_calls>",
        "|DSML|tool_calls>",
        "<tool_calls>",
        "<tool_call>",
        "<invoke ",
        "<|DSML|invoke ",
        "|DSML|invoke ",
    ]

    def __init__(self, parse_fn: Callable | None = None):
        self.parse_fn = parse_fn
        self._pending = ""
        self._capture_buf = ""
        self._capturing = False

    def feed(self, chunk: str) -> list[SieveEvent]:
        events: list[SieveEvent] = []

        if self._capturing:
            self._capture_buf += chunk
            result = self._try_finish_capture()
            if result is not None:
                prefix, tool_calls, suffix = result
                if prefix:
                    events.append(SieveEvent("text", prefix))
                if tool_calls:
                    events.append(SieveEvent("tool_calls", tool_calls))
                if suffix:
                    self._pending = suffix
                self._capture_buf = ""
                self._capturing = False
                if suffix:
                    events.extend(self.feed(""))
            return events

        self._pending += chunk
        start_idx = self._find_tool_start(self._pending)

        if start_idx >= 0:
            prefix = self._pending[:start_idx]
            rest = self._pending[start_idx:]
            self._pending = ""
            if prefix:
                events.append(SieveEvent("text", prefix))
            self._capture_buf = rest
            self._capturing = True
            result = self._try_finish_capture()
            if result is not None:
                prefix_text, tool_calls, suffix = result
                if prefix_text:
                    events.append(SieveEvent("text", prefix_text))
                if tool_calls:
                    events.append(SieveEvent("tool_calls", tool_calls))
                if suffix:
                    self._pending = suffix
                self._capture_buf = ""
                self._capturing = False
        else:
            safe, hold = self._split_safe(self._pending)
            if safe:
                events.append(SieveEvent("text", safe))
            self._pending = hold

        return events

    def flush(self) -> list[SieveEvent]:
        events: list[SieveEvent] = []
        if self._capturing:
            result = self._try_finish_capture()
            if result is not None:
                prefix, tool_calls, suffix = result
                if prefix:
                    events.append(SieveEvent("text", prefix))
                if tool_calls:
                    events.append(SieveEvent("tool_calls", tool_calls))
                if suffix:
                    events.append(SieveEvent("text", suffix))
            else:
                # 没闭合，当正文处理
                events.append(SieveEvent("text", self._capture_buf))
            self._capture_buf = ""
            self._capturing = False
        if self._pending:
            events.append(SieveEvent("text", self._pending))
            self._pending = ""
        return events

    def _find_tool_start(self, text: str) -> int:
        for tag in self._TOOL_STARTS:
            pos = text.find(tag)
            if pos >= 0:
                return pos
        for prefix in ("<|DSML|", "|DSML|", "<tool_calls", "<tool_call", "<invoke"):
            pos = text.find(prefix)
            if pos >= 0:
                return pos
        return -1

    def _split_safe(self, text: str) -> tuple[str, str]:
        last_lt = text.rfind("<")
        if last_lt == -1:
            last_lt = text.rfind("|")
        if last_lt == -1:
            return text, ""
        tail = text[last_lt:]
        for tag in self._TOOL_STARTS:
            if tag.startswith(tail) or tail == tag[:len(tail)]:
                return text[:last_lt], tail
        for prefix in ("<|DSML|", "|DSML|", "<tool_calls", "<tool_call", "<invoke"):
            if prefix.startswith(tail) or (len(tail) <= len(prefix) and tail == prefix[:len(tail)]):
                return text[:last_lt], tail
        return text, ""

    def _try_finish_capture(self):
        if not self._capture_buf or not self.parse_fn:
            return None
        if not self._is_capture_complete():
            return None
        tool_calls, cleaned = self.parse_fn(self._capture_buf)
        if tool_calls:
            return ("", tool_calls, "")
        return (self._capture_buf, None, "")

    def _is_capture_complete(self) -> bool:
        buf = self._capture_buf
        if "<|DSML|tool_calls>" in buf or "<tool_calls>" in buf:
            return "</|DSML|tool_calls>" in buf or "</tool_calls>" in buf
        if "<invoke " in buf or "<|DSML|invoke " in buf:
            return "</invoke>" in buf or "</|DSML|invoke>" in buf
        return False
