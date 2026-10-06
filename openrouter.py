"""OpenRouter backend: the same two-call turn as llm.ChatSession, over OpenRouter's OpenAI-style chat API.

Call 1 is a tool-calling loop that gathers facts and drafts the plan. Call 2 asks for that plan
as JSON matching `Itinerary`. Messages are plain dicts in the OpenAI chat format. The tools,
prompts, schema, error type and result type are shared with llm.py; only the wire format differs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import ValidationError

import llm
from prompts import FORMAT_PROMPT, system_prompt
from schemas import Itinerary
from tools import run_tool

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"  # override with OPENROUTER_MODEL in .env; it must support tool calling
REQUEST_TIMEOUT_S = 60    # longest silence from the server
REQUEST_DEADLINE_S = 90   # longest a single request may take in total, however busy the server keeps the line
RETRYABLE_STATUS = (429, 502, 503)


class OpenRouterError(Exception):
    """An HTTP-level failure from OpenRouter. `status` is the HTTP (or in-body) error code."""

    def __init__(self, status: int, message: str, retry_after: Optional[float] = None):
        super().__init__(f"OpenRouter {status}: {message}")
        self.status = status
        self.retry_after = retry_after


class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _post(self, payload: Dict[str, Any]) -> Any:
        """requests.post with a hard overall deadline.

        requests' own timeout only limits silence between bytes, and OpenRouter keeps a queued request
        alive with whitespace, so a busy free model could hang forever. The call runs in a daemon thread
        that we stop waiting for after REQUEST_DEADLINE_S.
        """
        outcome: Dict[str, Any] = {}

        def run() -> None:
            try:
                outcome["response"] = requests.post(
                    API_URL, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=REQUEST_TIMEOUT_S
                )
            except BaseException as exc:  # handed back to the caller's thread below
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(REQUEST_DEADLINE_S)
        if worker.is_alive():
            raise OpenRouterError(408, f"no answer within {REQUEST_DEADLINE_S:g} seconds")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["response"]

    def chat(self, **payload: Any) -> Dict[str, Any]:
        """POST one chat completion and return the parsed body. Failures raise OpenRouterError."""
        response = self._post(payload)
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if response.status_code >= 400 or error:
            # OpenRouter can also answer 200 with an error object in the body (e.g. the provider failed mid-request).
            status = response.status_code
            if status < 400:
                try:
                    status = int((error or {}).get("code"))
                except (TypeError, ValueError):
                    status = 502
            message = (error or {}).get("message") if isinstance(error, dict) else None
            try:
                retry_after = float(response.headers.get("Retry-After"))
            except (TypeError, ValueError):
                retry_after = None
            raise OpenRouterError(status, message or (response.text or "")[:200], retry_after)
        return body


# --- Tool declarations -------------------------------------------------------------

def _json_schema(schema: Any) -> Dict[str, Any]:
    """llm.TOOLS declares parameters as Gemini Schema objects; OpenRouter wants plain JSON Schema (lower-case types)."""
    def lower(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: (v.lower() if k == "type" and isinstance(v, str) else lower(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [lower(v) for v in node]
        return node
    return lower(schema.model_dump(mode="json", exclude_none=True))


TOOLS = [
    {"type": "function", "function": {"name": d.name, "description": d.description, "parameters": _json_schema(d.parameters)}}
    for tool in llm.TOOLS
    for d in tool.function_declarations
]


# --- Errors ------------------------------------------------------------------------

def friendly_error(exc: Exception, model: str = DEFAULT_MODEL) -> llm.ChatError:
    if isinstance(exc, llm.ChatError):
        return exc
    if isinstance(exc, OpenRouterError):
        status = exc.status
        if status == 401:
            return llm.ChatError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY in the .env file.")
        if status == 402:
            return llm.ChatError(
                f"Your OpenRouter account has no credit for {model!r}. Add credit at openrouter.ai, "
                "or set OPENROUTER_MODEL in the .env file to a model ending in :free."
            )
        if status == 403:
            return llm.ChatError("OpenRouter refused that request (it may have been flagged by the model's moderation). Try rephrasing it.")
        if status == 404:
            return llm.ChatError(
                f"OpenRouter found no provider for {model!r} that supports tool calling. Check OPENROUTER_MODEL in the .env file "
                "(for a :free model, also check your privacy settings at openrouter.ai/settings/privacy)."
            )
        if status == 408:
            return llm.ChatError("OpenRouter took too long to answer (free models can be slow when busy). Try again in a moment, or switch to a paid model.")
        if status == 429:
            return llm.ChatError("OpenRouter is rate-limiting requests (free models have tight limits). Wait a few seconds and try again.")
        if status >= 500:
            return llm.ChatError("OpenRouter or the model provider is temporarily unavailable. Try again in a moment.")
        return llm.ChatError("OpenRouter could not handle that request. Try rephrasing it.")
    if isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError)):
        return llm.ChatError("Could not reach OpenRouter. Check your internet connection and try again.")
    return llm.ChatError("Something went wrong while planning. Please try again.")


# --- Chat session ------------------------------------------------------------------

class OpenRouterSession:
    """One conversation. Same interface as llm.ChatSession; keeps the message history for follow-ups."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model
        self.messages: List[Dict[str, Any]] = []

    def send(self, user_text: str, on_event: Optional[llm.EventCallback] = None) -> llm.TurnResult:
        emit = on_event or (lambda kind, message: None)
        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        try:
            text, tool_calls = self._tool_loop(emit)
            itinerary = None
            if any(c.name == "check_budget" and "error" not in c.result for c in tool_calls):
                emit("format", "Building your itinerary…")
                itinerary = self._format_itinerary(text, emit)
            return llm.TurnResult(text=text, itinerary=itinerary, tool_calls=tool_calls)
        except Exception as exc:
            del self.messages[start:]  # leave the history as it was before this failed turn
            if not isinstance(exc, llm.ChatError):
                log.exception("Turn failed")
            raise friendly_error(exc, self.model) from exc

    def _complete(self, emit: llm.EventCallback, **payload: Any) -> Dict[str, Any]:
        """One chat completion, retried with a short wait when OpenRouter is rate-limiting or busy. Returns the message."""
        for attempt in range(len(llm.RETRY_DELAYS_S) + 1):
            try:
                body = self.client.chat(model=self.model, **payload)
                break
            except OpenRouterError as exc:
                if exc.status not in RETRYABLE_STATUS or attempt == len(llm.RETRY_DELAYS_S):
                    raise
                if exc.retry_after is not None and exc.retry_after > llm.MAX_SERVER_WAIT_S:
                    raise
                planned = llm.RETRY_DELAYS_S[attempt]
                delay = planned if exc.retry_after is None else max(planned, exc.retry_after + 1)
                log.warning("OpenRouter returned %s; retrying in %gs (attempt %d)", exc.status, delay, attempt + 1)
                emit("retry", f"OpenRouter is busy. Retrying in {delay:g}s…")
                time.sleep(delay)
        choices = body.get("choices") or []
        message = choices[0].get("message") if choices else None
        if not message:
            raise llm.ChatError("OpenRouter did not return an answer (the request may have been blocked). Try rephrasing it.")
        return message

    # Call 1: gather facts with tools, then draft the plan as text.
    def _tool_loop(self, emit: llm.EventCallback) -> Tuple[str, List[llm.ToolCall]]:
        calls_made: List[llm.ToolCall] = []
        for _ in range(llm.MAX_TOOL_ROUNDS):
            message = self._complete(
                emit,
                messages=[{"role": "system", "content": system_prompt()}, *self.messages],
                tools=TOOLS,
                tool_choice="auto",
            )
            self.messages.append(_assistant_entry(message))

            requested = message.get("tool_calls") or []
            if not requested:
                text = (message.get("content") or "").strip()
                if not text:
                    raise llm.ChatError("The model sent an empty reply. Please try again.")
                return text, calls_made

            for call in requested:
                function = call.get("function") or {}
                name = function.get("name") or ""
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except ValueError:
                    args = None
                valid = isinstance(args, dict)
                args = args if valid else {}
                emit("tool", llm._describe_call(name, args))
                result = run_tool(name, args) if valid else {"error": "The tool arguments were not a valid JSON object. Call the tool again with valid arguments."}
                emit("result", llm._describe_result(name, result))
                calls_made.append(llm.ToolCall(name, args, result))
                self.messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": json.dumps(result, default=str)})

        raise llm.ChatError("That request took too many steps. Try asking for a simpler trip.")

    # Call 2: the plan as JSON matching the Itinerary schema. Retry once, then give up (caller shows text).
    def _format_itinerary(self, draft: str, emit: llm.EventCallback) -> Optional[Itinerary]:
        prompt = (
            f"{FORMAT_PROMPT}\n\nJSON schema:\n{json.dumps(Itinerary.model_json_schema())}\n\n"
            f"Conversation:\n{self._transcript()}\n\nTool results:\n{self._tool_results()}\n\nPlan to convert:\n{draft}"
        )
        for attempt in range(llm.FORMAT_ATTEMPTS):
            message = self._complete(
                emit,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            try:
                return llm._reconcile(Itinerary.model_validate_json(_extract_json(message.get("content") or "")))
            except (ValidationError, ValueError, TypeError):
                log.warning("Itinerary JSON failed validation (attempt %d)", attempt + 1)
        return None

    def _transcript(self) -> str:
        lines = []
        for message in self.messages:
            text = (message.get("content") or "").strip() if message["role"] in ("user", "assistant") else ""
            if text:
                lines.append(f"{'User' if message['role'] == 'user' else 'Concierge'}: {text}")
        return "\n".join(lines)

    def _tool_results(self) -> str:
        names = {c.get("id"): (c.get("function") or {}).get("name") for m in self.messages for c in m.get("tool_calls") or []}
        lines = [f"{names.get(m.get('tool_call_id'), 'tool')}: {m['content']}" for m in self.messages if m["role"] == "tool"]
        return "\n".join(lines) or "(none)"


def _assistant_entry(message: Dict[str, Any]) -> Dict[str, Any]:
    """The model's reply as stored history: only the fields the API accepts back, plus reasoning details if it sent any."""
    entry: Dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        entry["tool_calls"] = message["tool_calls"]
    if message.get("reasoning_details"):
        entry["reasoning_details"] = message["reasoning_details"]
    return entry


def _extract_json(text: str) -> str:
    """The JSON object in a model reply, tolerating a ```json fence or a sentence around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def new_session() -> OpenRouterSession:
    llm.load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise llm.ChatError("No OpenRouter API key found. Add OPENROUTER_API_KEY to the .env file and restart the app.")
    return OpenRouterSession(OpenRouterClient(key), os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL)
