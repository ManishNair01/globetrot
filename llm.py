"""Gemini client, tool declarations, and the two-call turn: tool loop, then JSON formatting.

Call 1 is a tool-calling loop that gathers facts and drafts the plan. Call 2 asks
for that plan as JSON matching `Itinerary`. They are separate on purpose: tools
and strict JSON output in one call are not reliably supported across models.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from prompts import FORMAT_PROMPT, system_prompt
from schemas import Itinerary
from tools import check_budget, run_tool

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"  # override with GEMINI_MODEL in .env; model names change
MAX_TOOL_ROUNDS = 5
FORMAT_ATTEMPTS = 2  # first try plus one retry
RETRY_DELAYS_S = (2, 6)  # waits before retry 1 and 2 when Gemini says 429 (rate limit) or 503 (overloaded)
RETRYABLE_CODES = (429, 503)
MAX_SERVER_WAIT_S = 10  # if Gemini asks for a longer wait than this, retrying is pointless: fail fast

EventCallback = Callable[[str, str], None]  # (kind, message); kind is "tool", "result", "retry" or "format"


class ChatError(Exception):
    """An error whose message is safe to show to the user as-is."""


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class TurnResult:
    text: str
    itinerary: Optional[Itinerary] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


# --- Tool declarations -------------------------------------------------------

_S = types.Schema
_T = types.Type

TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the daily weather forecast (temperatures, rain chance, conditions) for a city and date range.",
            parameters=_S(
                type=_T.OBJECT,
                properties={
                    "city": _S(type=_T.STRING, description="A specific city or town, e.g. 'Jaipur'. Not a state or region."),
                    "start_date": _S(type=_T.STRING, description="First day of the trip, YYYY-MM-DD."),
                    "days": _S(type=_T.INTEGER, description="Number of trip days."),
                    "country_code": _S(type=_T.STRING, description="Optional ISO 3166-1 alpha-2 country code, e.g. 'IN', to pick the right place."),
                },
                required=["city", "start_date", "days"],
            ),
        ),
        types.FunctionDeclaration(
            name="check_budget",
            description="Add up the estimated cost of every activity and compare the total with the user's budget.",
            parameters=_S(
                type=_T.OBJECT,
                properties={
                    "activities": _S(
                        type=_T.ARRAY,
                        description="Every activity in the plan.",
                        items=_S(
                            type=_T.OBJECT,
                            properties={
                                "name": _S(type=_T.STRING),
                                "estimated_cost": _S(type=_T.NUMBER, description="Cost in the trip currency."),
                            },
                            required=["name", "estimated_cost"],
                        ),
                    ),
                    "budget": _S(type=_T.NUMBER, description="The user's total budget."),
                },
                required=["activities", "budget"],
            ),
        ),
        types.FunctionDeclaration(
            name="convert_currency",
            description="Convert an amount between two currencies at the latest exchange rate. Call it with amount 1 to get the rate, then multiply.",
            parameters=_S(
                type=_T.OBJECT,
                properties={
                    "amount": _S(type=_T.NUMBER, description="Amount to convert."),
                    "from_currency": _S(type=_T.STRING, description="3-letter currency code to convert from, e.g. 'EUR'."),
                    "to_currency": _S(type=_T.STRING, description="3-letter currency code to convert to, e.g. 'INR'."),
                },
                required=["amount", "from_currency", "to_currency"],
            ),
        ),
    ])
]


# --- Settings and error handling ---------------------------------------------

KEY_VARS = {"gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def provider() -> str:
    """Which backend to use: LLM_PROVIDER from .env, else OpenRouter if that is the only key present, else Gemini."""
    load_dotenv()
    choice = os.getenv("LLM_PROVIDER", "").strip().lower()
    if choice in KEY_VARS:
        return choice
    if choice:
        raise ChatError(f"LLM_PROVIDER must be 'gemini' or 'openrouter', not {choice!r}. Fix it in the .env file and restart the app.")
    if os.getenv("OPENROUTER_API_KEY", "").strip() and not os.getenv("GEMINI_API_KEY", "").strip():
        return "openrouter"
    return "gemini"


def has_api_key() -> bool:
    try:
        return bool(os.getenv(KEY_VARS[provider()], "").strip())
    except ChatError:
        return False


def missing_key_message() -> str:
    try:
        name = KEY_VARS[provider()]
    except ChatError as exc:
        return str(exc)
    return f"Add your {'OpenRouter' if name.startswith('OPENROUTER') else 'Gemini'} API key as {name} in the .env file, then restart the app."


def new_session() -> Any:
    """A chat session on the configured provider (see provider())."""
    if provider() == "openrouter":
        import openrouter  # imported here because openrouter.py imports this module
        return openrouter.new_session()
    return ChatSession(*make_client())


def make_client() -> Tuple[genai.Client, str]:
    if not has_api_key():
        raise ChatError("No Gemini API key found. Add GEMINI_API_KEY to the .env file and restart the app.")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip()), os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def friendly_error(exc: Exception, model: str = DEFAULT_MODEL) -> ChatError:
    if isinstance(exc, ChatError):
        return exc
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code == 429:
            return ChatError("Gemini is rate-limiting requests on the free tier. Wait a few seconds and try again.")
        if code in (401, 403):
            return ChatError("Gemini rejected the API key. Check GEMINI_API_KEY in the .env file.")
        if code == 404:
            return ChatError(f"Gemini could not find the model {model!r}. Check GEMINI_MODEL in the .env file.")
        if code and code >= 500:
            return ChatError("Gemini is temporarily unavailable. Try again in a moment.")
        return ChatError("Gemini could not handle that request. Try rephrasing it.")
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return ChatError("Could not reach Gemini. Check your internet connection and try again.")
    return ChatError("Something went wrong while planning. Please try again.")


# --- Chat session ------------------------------------------------------------

class ChatSession:
    """One conversation. Keeps the Gemini history so follow-ups ("make day 2 cheaper") have context."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model
        self.contents: List[types.Content] = []

    def send(self, user_text: str, on_event: Optional[EventCallback] = None) -> TurnResult:
        emit = on_event or (lambda kind, message: None)
        start = len(self.contents)
        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
        try:
            text, tool_calls = self._tool_loop(emit)
            itinerary = None
            if any(c.name == "check_budget" and "error" not in c.result for c in tool_calls):
                emit("format", "Building your itinerary…")
                itinerary = self._format_itinerary(text, emit)
            return TurnResult(text=text, itinerary=itinerary, tool_calls=tool_calls)
        except Exception as exc:
            del self.contents[start:]  # leave the history as it was before this failed turn
            if not isinstance(exc, ChatError):
                log.exception("Turn failed")
            raise friendly_error(exc, self.model) from exc

    def _generate(self, emit: EventCallback, **kwargs: Any) -> Any:
        """generate_content, retried a couple of times with a short wait when Gemini is rate-limiting or overloaded."""
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            try:
                return self.client.models.generate_content(model=self.model, **kwargs)
            except errors.APIError as exc:
                if getattr(exc, "code", None) not in RETRYABLE_CODES or attempt == len(RETRY_DELAYS_S):
                    raise
                asked = _server_wait_s(exc)
                if asked is not None and asked > MAX_SERVER_WAIT_S:
                    raise
                delay = RETRY_DELAYS_S[attempt] if asked is None else max(RETRY_DELAYS_S[attempt], asked + 1)
                log.warning("Gemini returned %s; retrying in %gs (attempt %d)", exc.code, delay, attempt + 1)
                emit("retry", f"Gemini is busy. Retrying in {delay:g}s…")
                time.sleep(delay)

    # Call 1: gather facts with tools, then draft the plan as text.
    def _tool_loop(self, emit: EventCallback) -> Tuple[str, List[ToolCall]]:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt(),
            tools=TOOLS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        calls_made: List[ToolCall] = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._generate(emit, contents=self.contents, config=config)
            content = _first_content(response)
            self.contents.append(content)  # keep as returned so any thought signatures survive

            requested = [p.function_call for p in (content.parts or []) if p.function_call]
            if not requested:
                text = "".join(p.text for p in (content.parts or []) if p.text and not p.thought).strip()
                if not text:
                    raise ChatError("Gemini sent an empty reply. Please try again.")
                return text, calls_made

            replies = []
            for request in requested:
                args = dict(request.args or {})
                emit("tool", _describe_call(request.name, args))
                result = run_tool(request.name, args)
                emit("result", _describe_result(request.name, result))
                calls_made.append(ToolCall(request.name, args, result))
                replies.append(types.Part.from_function_response(name=request.name, response={"result": result}))
            self.contents.append(types.Content(role="user", parts=replies))

        raise ChatError("That request took too many steps. Try asking for a simpler trip.")

    # Call 2: the plan as JSON matching the Itinerary schema. Retry once, then give up (caller shows text).
    def _format_itinerary(self, draft: str, emit: EventCallback) -> Optional[Itinerary]:
        prompt = (
            f"{FORMAT_PROMPT}\n\nConversation:\n{self._transcript()}\n\n"
            f"Tool results:\n{self._tool_results()}\n\nPlan to convert:\n{draft}"
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Itinerary, temperature=0.2
        )
        for attempt in range(FORMAT_ATTEMPTS):
            response = self._generate(emit, contents=prompt, config=config)
            try:
                return _reconcile(Itinerary.model_validate_json(response.text))
            except (ValidationError, ValueError, TypeError):
                log.warning("Itinerary JSON failed validation (attempt %d)", attempt + 1)
        return None

    def _transcript(self) -> str:
        lines = []
        for content in self.contents:
            text = "".join(p.text for p in (content.parts or []) if p.text and not p.thought).strip()
            if text:
                lines.append(f"{'User' if content.role == 'user' else 'Concierge'}: {text}")
        return "\n".join(lines)

    def _tool_results(self) -> str:
        lines = []
        for content in self.contents:
            for part in content.parts or []:
                if part.function_response:
                    lines.append(f"{part.function_response.name}: {json.dumps(part.function_response.response, default=str)}")
        return "\n".join(lines) or "(none)"


def _server_wait_s(exc: errors.APIError) -> Optional[float]:
    """The wait Gemini asks for in a 429's RetryInfo detail (e.g. "7s"), if it gave one."""
    try:
        for detail in exc.details["error"]["details"]:
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                return float(str(detail["retryDelay"]).rstrip("s"))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return None


def _first_content(response: Any) -> types.Content:
    candidates = getattr(response, "candidates", None) or []
    if not candidates or candidates[0].content is None:
        raise ChatError("Gemini did not return an answer (the request may have been blocked). Try rephrasing it.")
    return candidates[0].content


def _reconcile(itinerary: Itinerary) -> Itinerary:
    """Trust the arithmetic in Python, not the model: recompute the total and the budget status."""
    activities = [{"name": a.name, "estimated_cost": a.estimated_cost} for d in itinerary.days for a in d.activities]
    check = check_budget(activities, itinerary.budget)
    if "error" in check:
        return itinerary
    return itinerary.model_copy(update={"estimated_total": check["total"], "budget_status": check["status"]})


def _describe_call(name: str, args: Dict[str, Any]) -> str:
    if name == "get_weather":
        return f"Checking weather for {args.get('city', 'your destination')}…"
    if name == "check_budget":
        return "Checking the plan against your budget…"
    if name == "convert_currency":
        return f"Converting {args.get('from_currency', '?')} to {args.get('to_currency', '?')}…"
    return f"Running {name}…"


def _describe_result(name: str, result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"Note: {result['error']}"
    if name == "get_weather":
        return f"Matched {result.get('place', 'the destination')} · {len(result.get('forecast', []))} days of forecast"
    if name == "check_budget":
        return f"Total {result['total']:,.0f} of {result['budget']:,.0f} · {result['status']} budget"
    if name == "convert_currency":
        when = f" · {result['rate_date']}" if result.get("rate_date") else ""
        return f"1 {result['from']} = {result['rate']:,.4g} {result['to']}{when}"
    return "Done"
