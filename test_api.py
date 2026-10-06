"""Offline tests for the HTTP API, the journey prefill and tool logging. No network, and the real .env is never read.

Run with:  python -m unittest test_api -v
"""
from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import api
import content
import llm
import tools
from schemas import Itinerary
from test_chatbot import ITINERARY_JSON


class FakeSession:
    """Stands in for a provider session: records what it was sent, emits one event, replies with a canned result."""

    def __init__(self, result=None, error=None):
        self.sent = []
        self.result = result if result is not None else llm.TurnResult(text="Hello from the concierge")
        self.error = error

    def send(self, text, on_event=None):
        self.sent.append(text)
        if on_event:
            on_event("tool", "Checking weather for Goa…")
        if self.error:
            raise self.error
        return self.result


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.contact_file = Path(self.tmp.name) / "contact.jsonl"
        self.sessions = []
        self.clock = Clock()

    def make_client(self, factory=None, site_url="http://localhost:5173", **kwargs):
        def default_factory():
            session = FakeSession()
            self.sessions.append(session)
            return session

        store = api.SessionStore(factory or default_factory, clock=self.clock)
        app = api.create_app(site_url=site_url, contact_file=self.contact_file, store=store, **kwargs)
        return TestClient(app)


class ChatTests(ApiTestCase):
    def test_a_turn_returns_text_events_and_a_session_id(self):
        body = self.make_client().post("/api/chat", json={"message": "3 days in Goa"}).json()
        self.assertEqual(body["text"], "Hello from the concierge")
        self.assertEqual(body["events"], [{"kind": "tool", "text": "Checking weather for Goa…"}])
        self.assertTrue(body["session_id"])
        self.assertIsNone(body["error"])
        self.assertIsNone(body["itinerary"])

    def test_an_itinerary_is_returned_as_json(self):
        itinerary = Itinerary.model_validate_json(ITINERARY_JSON)
        factory = lambda: FakeSession(llm.TurnResult(text="Plan", itinerary=itinerary))
        body = self.make_client(factory).post("/api/chat", json={"message": "plan"}).json()
        self.assertEqual(body["itinerary"]["destination"], "Goa, India")
        self.assertEqual(body["itinerary"]["days"][0]["activities"][0]["time_of_day"], itinerary.days[0].activities[0].time_of_day)

    def test_the_same_session_id_keeps_the_conversation(self):
        client = self.make_client()
        first = client.post("/api/chat", json={"message": "3 days in Goa"}).json()
        client.post("/api/chat", json={"session_id": first["session_id"], "message": "make day 2 cheaper"})
        self.assertEqual(len(self.sessions), 1)
        self.assertEqual(self.sessions[0].sent, ["3 days in Goa", "make day 2 cheaper"])

    def test_an_unknown_session_id_gets_a_fresh_session_with_a_new_id(self):
        client = self.make_client()
        body = client.post("/api/chat", json={"session_id": "made-up", "message": "hi"}).json()
        self.assertNotEqual(body["session_id"], "made-up")
        self.assertEqual(len(self.sessions), 1)

    def test_an_idle_session_expires(self):
        client = self.make_client()
        first = client.post("/api/chat", json={"message": "hi"}).json()
        self.clock.now += api.SESSION_TTL_S + 1
        second = client.post("/api/chat", json={"session_id": first["session_id"], "message": "hi again"}).json()
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertEqual(len(self.sessions), 2)

    def test_reset_drops_the_session(self):
        client = self.make_client()
        first = client.post("/api/chat", json={"message": "hi"}).json()
        self.assertEqual(client.post("/api/chat/reset", json={"session_id": first["session_id"]}).json(), {"ok": True})
        second = client.post("/api/chat", json={"session_id": first["session_id"], "message": "hi"}).json()
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_reset_of_an_unknown_session_is_fine(self):
        self.assertEqual(self.make_client().post("/api/chat/reset", json={"session_id": "nope"}).json(), {"ok": True})
        self.assertEqual(self.make_client().post("/api/chat/reset", json={}).json(), {"ok": True})

    def test_a_chat_error_is_reported_in_the_body_not_as_a_crash(self):
        factory = lambda: FakeSession(error=llm.ChatError("Gemini is rate-limiting requests."))
        response = self.make_client(factory).post("/api/chat", json={"message": "hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "Gemini is rate-limiting requests.")
        self.assertEqual(response.json()["events"][0]["kind"], "tool")  # what happened before the failure is kept

    def test_a_missing_api_key_is_reported_and_stores_no_session(self):
        def no_key():
            raise llm.ChatError("Add your OpenRouter API key as OPENROUTER_API_KEY in the .env file, then restart the app.")

        client = self.make_client(no_key)
        body = client.post("/api/chat", json={"message": "hi"}).json()
        self.assertIn("OPENROUTER_API_KEY", body["error"])
        self.assertIsNone(body["session_id"])

    def test_an_unexpected_exception_does_not_leak_details(self):
        factory = lambda: FakeSession(error=RuntimeError("secret internals sk-or-v1-abc"))
        with self.assertLogs("api", level="ERROR"):
            body = self.make_client(factory).post("/api/chat", json={"message": "hi"}).json()
        self.assertNotIn("sk-or", json.dumps(body))
        self.assertIn("Something went wrong", body["error"])

    def test_bad_bodies_are_rejected(self):
        client = self.make_client()
        for payload in ({}, {"message": ""}, {"message": "   "}, {"message": "x" * (api.MAX_MESSAGE_CHARS + 1)}, {"message": 5}):
            self.assertEqual(client.post("/api/chat", json=payload).status_code, 422, payload)

    def test_chat_is_rate_limited_per_client_with_retry_after(self):
        limiter = api.RateLimiter(2, 60, clock=self.clock)
        client = self.make_client(chat_limiter=limiter)
        self.assertEqual([client.post("/api/chat", json={"message": "hi"}).status_code for _ in range(2)], [200, 200])
        blocked = client.post("/api/chat", json={"message": "hi"})
        self.assertEqual(blocked.status_code, 429)
        self.assertGreaterEqual(int(blocked.headers["Retry-After"]), 1)
        self.clock.now += 61
        self.assertEqual(client.post("/api/chat", json={"message": "hi"}).status_code, 200)


class RateLimiterTests(unittest.TestCase):
    def test_window_slides_and_keys_are_independent(self):
        clock = Clock()
        limiter = api.RateLimiter(2, 10, clock=clock)
        self.assertIsNone(limiter.check("a"))
        self.assertIsNone(limiter.check("a"))
        self.assertAlmostEqual(limiter.check("a"), 10)
        self.assertIsNone(limiter.check("b"))
        clock.now += 6
        self.assertAlmostEqual(limiter.check("a"), 4)  # a blocked hit is not recorded, so the wait keeps shrinking
        clock.now += 5
        self.assertIsNone(limiter.check("a"))

    def test_idle_keys_are_forgotten(self):
        clock = Clock()
        limiter = api.RateLimiter(1, 10, clock=clock)
        limiter.check("a")
        clock.now += 11
        limiter.check("b")
        self.assertEqual(set(limiter._hits), {"b"})


class SessionStoreTests(unittest.TestCase):
    def test_the_least_recently_used_session_is_evicted_when_full(self):
        clock = Clock()
        store = api.SessionStore(object, max_sessions=2, clock=clock)
        first, _ = store.acquire(None)
        clock.now += 1
        second, _ = store.acquire(None)
        clock.now += 1
        store.acquire(first)  # touch the first, so the second is now the oldest
        clock.now += 1
        third, _ = store.acquire(None)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.acquire(first)[0], first)
        self.assertNotEqual(store.acquire(second)[0], second)
        self.assertNotIn(third, [None])


class ReadOnlyRouteTests(ApiTestCase):
    def test_health_never_includes_the_key(self):
        env = {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-secret", "OPENROUTER_MODEL": "some/model"}
        with mock.patch.dict("os.environ", env, clear=True), mock.patch("llm.load_dotenv"), mock.patch("api.load_dotenv"):
            response = self.make_client().get("/api/health")
        self.assertEqual(response.json(), {"ok": True, "provider": "openrouter", "model": "some/model", "has_key": True})
        self.assertNotIn("sk-or", response.text)

    def test_health_reports_a_bad_provider_setting(self):
        with mock.patch.dict("os.environ", {"LLM_PROVIDER": "bogus"}, clear=True), mock.patch("llm.load_dotenv"), mock.patch("api.load_dotenv"):
            body = self.make_client().get("/api/health").json()
        self.assertFalse(body["ok"])
        self.assertIn("LLM_PROVIDER", body["error"])

    def test_journeys_match_the_content_store(self):
        body = self.make_client().get("/api/journeys").json()
        self.assertEqual(body, content.load_journeys())
        self.assertIn("goa", [j["slug"] for j in body["journeys"]])

    def test_cors_allows_only_the_site_origin(self):
        client = self.make_client(site_url="http://localhost:5173/some/path")
        allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "http://localhost:5173")
        other = client.get("/api/health", headers={"Origin": "https://evil.example"})
        self.assertNotIn("access-control-allow-origin", other.headers)


class ContactTests(ApiTestCase):
    GOOD = {"name": "Asha", "email": "asha@example.com", "message": "Can you plan Kerala?"}

    def test_a_message_is_saved_as_one_json_line(self):
        client = self.make_client()
        self.assertEqual(client.post("/api/contact", json=self.GOOD).json(), {"ok": True})
        client.post("/api/contact", json={**self.GOOD, "name": "Ravi"})
        lines = self.contact_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        saved = json.loads(lines[0])
        self.assertEqual((saved["name"], saved["email"], saved["message"]), ("Asha", "asha@example.com", "Can you plan Kerala?"))
        self.assertIn("received_at", saved)

    def test_text_with_newlines_and_unicode_stays_on_one_line(self):
        self.make_client().post("/api/contact", json={**self.GOOD, "message": "line one\nline two ₹5,000"})
        lines = self.contact_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["message"], "line one\nline two ₹5,000")

    def test_bad_input_is_rejected_and_nothing_is_saved(self):
        client = self.make_client()
        for payload in ({}, {**self.GOOD, "email": "not-an-email"}, {**self.GOOD, "name": "  "}, {**self.GOOD, "message": ""},
                        {**self.GOOD, "message": "x" * (api.MAX_MESSAGE_CHARS + 1)}):
            self.assertEqual(client.post("/api/contact", json=payload).status_code, 422, payload)
        self.assertFalse(self.contact_file.exists())

    def test_contact_is_rate_limited(self):
        client = self.make_client(contact_limiter=api.RateLimiter(1, 60, clock=self.clock))
        self.assertEqual(client.post("/api/contact", json=self.GOOD).status_code, 200)
        self.assertEqual(client.post("/api/contact", json=self.GOOD).status_code, 429)

    def test_an_unwritable_location_is_a_clean_500(self):
        blocker = Path(self.tmp.name) / "file"
        blocker.write_text("x")
        app = api.create_app(contact_file=blocker / "contact.jsonl", store=api.SessionStore(FakeSession))
        with self.assertLogs("api", level="ERROR"):
            response = TestClient(app).post("/api/contact", json=self.GOOD)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(str(blocker), response.text)


class JourneyPrefillTests(unittest.TestCase):
    def test_a_known_journey_gives_an_opening_message(self):
        text = content.journey_prompt("goa")
        self.assertIn("Goa, Unhurried", text)
        self.assertIn("4 days", text)
        self.assertIn("₹20,000", text)

    def test_a_journey_without_a_budget_does_not_invent_one(self):
        text = content.journey_prompt("paris")
        self.assertIn("7 days", text)
        self.assertNotIn("about", text)

    def test_unknown_or_malformed_slugs_give_nothing(self):
        for slug in ("nowhere", "", None, "../etc/passwd", "GOA", "a" * 100, 5, ["goa"]):
            self.assertIsNone(content.journey_prompt(slug), slug)

    def test_every_journey_page_links_its_own_slug(self):
        for journey in content.load_journeys()["journeys"]:
            page = (content.JOURNEYS_FILE.parent / "journeys" / journey["slug"] / "index.html").read_text(encoding="utf-8")
            self.assertIn(f'data-chat-journey="{journey["slug"]}"', page)


class ToolLoggingTests(unittest.TestCase):
    def test_tool_calls_are_logged_without_arguments(self):
        with self.assertLogs("tools", level="INFO") as logged:
            tools.run_tool("check_budget", {"activities": [{"name": "Dinner", "estimated_cost": 500}], "budget": 1000})
            tools.run_tool("nope", {})
        text = "\n".join(logged.output)
        self.assertIn("tool check_budget: ok", text)
        self.assertIn("tool nope: unknown", text)
        self.assertNotIn("Dinner", text)


if __name__ == "__main__":
    unittest.main()
