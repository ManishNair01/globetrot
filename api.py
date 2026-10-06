"""HTTP API in front of the concierge, for pages that cannot use Streamlit (backend-skeleton.md §4 "Option B").

It adds no planning logic: a chat turn is `llm.new_session().send(...)`, exactly what the Streamlit page runs.
Sessions live in memory with an idle TTL (no database), and every route is rate-limited per client.

    uvicorn api:app --port 8000        (or the "api" entry in .claude/launch.json)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import content
import llm
import logs

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
CHAT_LIMIT, CHAT_WINDOW_S = 10, 60            # chat turns per client per minute (each one costs model calls)
CONTACT_LIMIT, CONTACT_WINDOW_S = 5, 600      # contact messages per client per ten minutes
SESSION_TTL_S = 30 * 60                        # a session idle this long is dropped
MAX_SESSIONS = 200                             # beyond this the least recently used session is dropped
MAX_MESSAGE_CHARS = 2000
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


# --- Rate limit and session store (plain classes so tests can drive the clock) -----------

class RateLimiter:
    """At most `limit` hits per `window_s` seconds for each key (sliding window)."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Optional[float]:
        """Record a hit and return None, or return the seconds to wait if `key` is over the limit (nothing recorded)."""
        now = self.clock()
        with self._lock:
            for stale in [k for k, hits in self._hits.items() if not hits or now - hits[-1] >= self.window_s]:
                del self._hits[stale]
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window_s - (now - hits[0])
            hits.append(now)
            return None


class _Entry:
    def __init__(self, session: Any, now: float):
        self.session = session
        self.lock = threading.Lock()  # one turn at a time per conversation
        self.last_used = now


class SessionStore:
    """Conversations by server-issued id. Expired ones are swept on access; the oldest is evicted when full."""

    def __init__(self, factory: Callable[[], Any], ttl_s: float = SESSION_TTL_S, max_sessions: int = MAX_SESSIONS,
                 clock: Callable[[], float] = time.monotonic):
        self.factory, self.ttl_s, self.max_sessions, self.clock = factory, ttl_s, max_sessions, clock
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def acquire(self, session_id: Optional[str]) -> Tuple[str, _Entry]:
        """The entry for `session_id`, or a new one (with a new id) if it is missing, unknown or expired.

        The factory may raise llm.ChatError (e.g. no API key); nothing is stored in that case.
        """
        now = self.clock()
        with self._lock:
            for stale in [i for i, e in self._entries.items() if now - e.last_used >= self.ttl_s]:
                del self._entries[stale]
            entry = self._entries.get(session_id) if session_id else None
            if entry is None:
                entry = _Entry(self.factory(), now)
                session_id = secrets.token_urlsafe(16)
                while len(self._entries) >= self.max_sessions:
                    del self._entries[min(self._entries, key=lambda i: self._entries[i].last_used)]
                self._entries[session_id] = entry
            entry.last_used = now
            return session_id, entry  # type: ignore[return-value]

    def drop(self, session_id: Optional[str]) -> None:
        with self._lock:
            self._entries.pop(session_id or "", None)


# --- Request and response bodies ----------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    text: str = ""
    itinerary: Optional[Dict[str, Any]] = None
    events: List[Dict[str, str]] = []
    error: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=64)


class ContactRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=200)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("name", "message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("email")
    @classmethod
    def _looks_like_email(cls, value: str) -> str:
        value = value.strip()
        if not _EMAIL.fullmatch(value):
            raise ValueError("not an email address")
        return value


# --- App ----------------------------------------------------------------------------------

def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _rate_limited(wait: float) -> HTTPException:
    seconds = max(1, math.ceil(wait))
    return HTTPException(429, f"Too many requests. Try again in {seconds} seconds.", headers={"Retry-After": str(seconds)})


def create_app(
    session_factory: Optional[Callable[[], Any]] = None,
    contact_file: Optional[Path] = None,
    site_url: Optional[str] = None,
    chat_limiter: Optional[RateLimiter] = None,
    contact_limiter: Optional[RateLimiter] = None,
    store: Optional[SessionStore] = None,
) -> FastAPI:
    load_dotenv()
    logs.configure()
    site_url = site_url or os.getenv("SITE_URL", "http://localhost:5173")
    contact_path = contact_file or Path(os.getenv("CONTACT_FILE") or ROOT / "data" / "contact_messages.jsonl")
    # `is None`, not `or`: an empty SessionStore has len 0 and would count as falsy.
    chat_limiter = RateLimiter(CHAT_LIMIT, CHAT_WINDOW_S) if chat_limiter is None else chat_limiter
    contact_limiter = RateLimiter(CONTACT_LIMIT, CONTACT_WINDOW_S) if contact_limiter is None else contact_limiter
    store = SessionStore(session_factory or llm.new_session) if store is None else store
    contact_lock = threading.Lock()

    app = FastAPI(title="GlobeTrot API", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=[_origin(site_url)], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def client_key(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        try:
            name = llm.provider()
        except llm.ChatError as exc:
            return {"ok": False, "provider": None, "model": None, "has_key": False, "error": str(exc)}
        model = os.getenv("OPENROUTER_MODEL" if name == "openrouter" else "GEMINI_MODEL", "").strip() or None
        return {"ok": True, "provider": name, "model": model, "has_key": llm.has_api_key()}

    @app.get("/api/journeys")
    def journeys() -> Dict[str, Any]:
        return content.load_journeys()

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        wait = chat_limiter.check(client_key(request))
        if wait is not None:
            raise _rate_limited(wait)
        try:
            session_id, entry = store.acquire(body.session_id)
        except llm.ChatError as exc:
            return ChatResponse(error=str(exc))
        events: List[Dict[str, str]] = []
        with entry.lock:
            try:
                result = entry.session.send(body.message, lambda kind, text: events.append({"kind": kind, "text": text}))
            except llm.ChatError as exc:
                return ChatResponse(session_id=session_id, events=events, error=str(exc))
            except Exception:  # sessions map their own failures to ChatError; this is the last line of defence
                log.exception("Unexpected failure in a chat turn")
                return ChatResponse(session_id=session_id, events=events, error="Something went wrong while planning. Please try again.")
        itinerary = result.itinerary.model_dump(mode="json") if result.itinerary else None
        return ChatResponse(session_id=session_id, text=result.text, itinerary=itinerary, events=events)

    @app.post("/api/chat/reset")
    def reset(body: ResetRequest) -> Dict[str, bool]:
        store.drop(body.session_id)
        return {"ok": True}

    @app.post("/api/contact")
    def contact(body: ContactRequest, request: Request) -> Dict[str, bool]:
        wait = contact_limiter.check(client_key(request))
        if wait is not None:
            raise _rate_limited(wait)
        record = {"received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **body.model_dump()}
        try:
            with contact_lock:
                contact_path.parent.mkdir(parents=True, exist_ok=True)
                with contact_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("Could not save a contact message")
            raise HTTPException(500, "Your message could not be saved. Please email us instead.")
        log.info("contact message saved")
        return {"ok": True}

    return app


app = create_app()
