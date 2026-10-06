"""Offline tests for the OpenRouter backend and provider selection. No network, and the real .env is never read.

Run with:  python -m unittest test_openrouter -v
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import threading
import time
import unittest
from unittest import mock

import requests

import llm
import openrouter
import tools
from openrouter import OpenRouterClient, OpenRouterError, OpenRouterSession
from test_chatbot import ITINERARY_JSON


# --- Fakes -------------------------------------------------------------------------

def _tool_call(call_id, name, **args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _reply(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


class FakeClient:
    """Returns queued bodies in order (or raises queued exceptions) and records every request."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.requests = []

    def chat(self, **payload):
        self.requests.append(copy.deepcopy(payload))
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fake_tools(name, args):
    return tools.check_budget(**args) if name == "check_budget" else {"place": "Goa, India", "forecast": [{}]}


# --- Session -----------------------------------------------------------------------

class SessionTests(unittest.TestCase):
    def test_planning_turn_runs_tools_then_formats(self):
        beach_and_museum = [{"name": "Beach", "estimated_cost": 700}, {"name": "Museum", "estimated_cost": 600}]
        client = FakeClient(
            _reply(tool_calls=[_tool_call("c1", "get_weather", city="Goa", start_date="2026-10-03", days=1)]),
            _reply(tool_calls=[_tool_call("c2", "check_budget", activities=beach_and_museum, budget=1000)]),
            _reply("Day 1: beach, museum."),
            _reply(ITINERARY_JSON),
        )
        events = []
        with mock.patch("openrouter.run_tool", side_effect=_fake_tools):
            result = OpenRouterSession(client, "some/model").send("1 day in Goa, 1000", lambda k, m: events.append((k, m)))

        self.assertEqual([c.name for c in result.tool_calls], ["get_weather", "check_budget"])
        # The model claimed total 1 / within; Python recomputes 1300 / over.
        self.assertEqual((result.itinerary.estimated_total, result.itinerary.budget_status), (1300, "over"))
        self.assertEqual(result.text, "Day 1: beach, museum.")
        self.assertIn("Checking weather for Goa…", [m for _, m in events])
        self.assertEqual([k for k, _ in events][-1], "format")

        first = client.requests[0]
        self.assertEqual((first["model"], first["tool_choice"], first["tools"]), ("some/model", "auto", openrouter.TOOLS))
        self.assertEqual([m["role"] for m in first["messages"]], ["system", "user"])

        second = client.requests[1]["messages"]  # the tool result goes back under the id the model gave
        self.assertEqual([m["role"] for m in second], ["system", "user", "assistant", "tool"])
        self.assertEqual(second[2]["tool_calls"][0]["id"], "c1")
        self.assertEqual(second[3]["tool_call_id"], "c1")
        self.assertEqual(json.loads(second[3]["content"])["place"], "Goa, India")

        formatting = client.requests[3]
        self.assertEqual(formatting["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", formatting)
        prompt = formatting["messages"][0]["content"]
        self.assertIn("Plan to convert:\nDay 1: beach, museum.", prompt)
        self.assertIn("get_weather: ", prompt)  # tool results are named, not just listed
        self.assertIn('"properties"', prompt)   # the schema is in the prompt: json_object mode does not enforce one

    def test_question_turn_makes_no_tool_call_and_no_itinerary(self):
        client = FakeClient(_reply("Where would you like to go, and for how long?"))
        result = OpenRouterSession(client, "m").send("plan me a trip")
        self.assertIsNone(result.itinerary)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(len(client.requests), 1)

    def test_bad_json_retries_once_then_falls_back_to_text(self):
        client = FakeClient(
            _reply(tool_calls=[_tool_call("c1", "check_budget", activities=[{"name": "a", "estimated_cost": 1}], budget=10)]),
            _reply("Draft plan"),
            _reply("not json"), _reply("{}"),
        )
        result = OpenRouterSession(client, "m").send("go")
        self.assertIsNone(result.itinerary)
        self.assertEqual(result.text, "Draft plan")
        self.assertEqual(len(client.requests), 4)  # 2 loop calls + 2 format attempts

    def test_json_in_a_code_fence_or_a_sentence_is_accepted(self):
        for wrapped in (f"```json\n{ITINERARY_JSON}\n```", f"Here is the itinerary:\n{ITINERARY_JSON}\nHope that helps!"):
            client = FakeClient(
                _reply(tool_calls=[_tool_call("c1", "check_budget", activities=[{"name": "a", "estimated_cost": 1}], budget=10)]),
                _reply("Draft plan"),
                _reply(wrapped),
            )
            self.assertIsNotNone(OpenRouterSession(client, "m").send("go").itinerary, wrapped[:20])

    def test_tool_loop_is_capped(self):
        looping = [_reply(tool_calls=[_tool_call(f"c{i}", "get_weather", city="Goa", start_date="2026-10-03", days=1)]) for i in range(llm.MAX_TOOL_ROUNDS)]
        session = OpenRouterSession(FakeClient(*looping), "m")
        with mock.patch("openrouter.run_tool", return_value={"forecast": []}), self.assertRaises(llm.ChatError):
            session.send("go")
        self.assertEqual(session.messages, [])  # a failed turn leaves no history behind

    def test_unreadable_tool_arguments_become_an_error_result_not_a_crash(self):
        broken = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{not json"}}
        client = FakeClient(_reply(tool_calls=[broken]), _reply("Sorry, which city?"))
        with mock.patch("openrouter.run_tool") as run:
            result = OpenRouterSession(client, "m").send("go")
        run.assert_not_called()
        self.assertEqual(result.text, "Sorry, which city?")
        sent_back = client.requests[1]["messages"][-1]
        self.assertEqual(sent_back["tool_call_id"], "c1")
        self.assertIn("valid JSON", json.loads(sent_back["content"])["error"])

    def test_history_carries_into_follow_up(self):
        client = FakeClient(_reply("Sure."), _reply("Done."))
        session = OpenRouterSession(client, "m")
        session.send("first")
        session.send("second")
        self.assertEqual([m["role"] for m in client.requests[1]["messages"]], ["system", "user", "assistant", "user"])

    def test_reasoning_details_are_passed_back_to_the_model(self):
        reply = _reply(tool_calls=[_tool_call("c1", "get_weather", city="Goa", start_date="2026-10-03", days=1)])
        reply["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.text", "text": "hmm"}]
        client = FakeClient(reply, _reply("Done."))
        with mock.patch("openrouter.run_tool", return_value={"forecast": []}):
            OpenRouterSession(client, "m").send("go")
        self.assertEqual(client.requests[1]["messages"][2]["reasoning_details"], [{"type": "reasoning.text", "text": "hmm"}])

    def test_empty_and_missing_replies_are_friendly_errors(self):
        with self.assertRaisesRegex(llm.ChatError, "empty reply"):
            OpenRouterSession(FakeClient(_reply("   ")), "m").send("hi")
        with self.assertRaisesRegex(llm.ChatError, "did not return an answer"):
            OpenRouterSession(FakeClient({"choices": []}), "m").send("hi")

    def test_errors_become_friendly_messages(self):
        cases = [
            (401, "rejected the API key"), (402, "no credit"), (403, "refused"), (404, "no provider"),
            (408, "too long"), (500, "temporarily unavailable"), (400, "Try rephrasing"),
        ]
        for status, message in cases:
            session = OpenRouterSession(FakeClient(OpenRouterError(status, "x")), "some/model")
            with self.assertRaisesRegex(llm.ChatError, message):
                session.send("hi")
            self.assertEqual(session.messages, [], status)

    def test_model_name_is_in_the_credit_and_provider_errors(self):
        for status in (402, 404):
            with self.assertRaisesRegex(llm.ChatError, "some/model"):
                OpenRouterSession(FakeClient(OpenRouterError(status, "x")), "some/model").send("hi")

    def test_offline_becomes_friendly_message(self):
        with self.assertRaisesRegex(llm.ChatError, "internet"):
            OpenRouterSession(FakeClient(requests.ConnectionError("no route")), "m").send("hi")
        with self.assertRaisesRegex(llm.ChatError, "internet"):
            OpenRouterSession(FakeClient(requests.Timeout("slow")), "m").send("hi")


class RetryTests(unittest.TestCase):
    def test_rate_limit_is_retried_then_succeeds(self):
        client = FakeClient(OpenRouterError(429, "x"), OpenRouterError(503, "x"), _reply("Where to?"))
        events = []
        with mock.patch("openrouter.time.sleep") as sleep:
            result = OpenRouterSession(client, "m").send("hi", lambda k, m: events.append((k, m)))
        self.assertEqual(result.text, "Where to?")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 6])
        self.assertEqual(events, [("retry", "OpenRouter is busy. Retrying in 2s…"), ("retry", "OpenRouter is busy. Retrying in 6s…")])

    def test_rate_limit_gives_up_after_the_last_retry(self):
        client = FakeClient(*[OpenRouterError(429, "x")] * 3)
        session = OpenRouterSession(client, "m")
        with mock.patch("openrouter.time.sleep") as sleep, self.assertRaisesRegex(llm.ChatError, "rate-limiting"):
            session.send("hi")
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(client.queue, [])  # all three attempts were used
        self.assertEqual(session.messages, [])

    def test_errors_that_retrying_cannot_fix_are_not_retried(self):
        for status in (401, 402, 404):
            client = FakeClient(OpenRouterError(status, "x"), _reply("never reached"))
            with mock.patch("openrouter.time.sleep") as sleep, self.assertRaises(llm.ChatError):
                OpenRouterSession(client, "m").send("hi")
            sleep.assert_not_called()
            self.assertEqual(len(client.queue), 1, status)

    def test_wait_requested_by_openrouter_is_honoured(self):
        client = FakeClient(OpenRouterError(429, "x", retry_after=7), _reply("ok"))
        with mock.patch("openrouter.time.sleep") as sleep:
            OpenRouterSession(client, "m").send("hi")
        sleep.assert_called_once_with(8)  # the 7s asked for, plus a second of margin

    def test_long_wait_requested_by_openrouter_fails_fast(self):
        client = FakeClient(OpenRouterError(429, "x", retry_after=34), _reply("never reached"))
        with mock.patch("openrouter.time.sleep") as sleep, self.assertRaisesRegex(llm.ChatError, "rate-limiting"):
            OpenRouterSession(client, "m").send("hi")
        sleep.assert_not_called()

    def test_formatting_call_is_retried_too(self):
        client = FakeClient(
            _reply(tool_calls=[_tool_call("c1", "check_budget", activities=[{"name": "Beach", "estimated_cost": 700}], budget=1000)]),
            _reply("Draft plan"),
            OpenRouterError(429, "x"),
            _reply(ITINERARY_JSON),
        )
        with mock.patch("openrouter.time.sleep"):
            result = OpenRouterSession(client, "m").send("go")
        self.assertIsNotNone(result.itinerary)


# --- HTTP client -------------------------------------------------------------------

def _http(body, status=200, headers=None, text=None):
    response = mock.Mock()
    response.status_code = status
    response.headers = headers or {}
    response.text = text if text is not None else json.dumps(body)
    if body is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = body
    return response


class ClientTests(unittest.TestCase):
    KEY = "sk-or-v1-test-key-123"

    def chat(self, response, **payload):
        with mock.patch("openrouter.requests.post", return_value=response) as post:
            return OpenRouterClient(self.KEY).chat(**payload), post

    def test_sends_the_request_and_returns_the_body(self):
        body, post = self.chat(_http({"choices": [{"message": {"content": "hi"}}]}), model="m", messages=[])
        self.assertEqual(body["choices"][0]["message"]["content"], "hi")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(kwargs["json"], {"model": "m", "messages": []})
        self.assertEqual(kwargs["headers"], {"Authorization": f"Bearer {self.KEY}"})
        self.assertGreater(kwargs["timeout"], 0)

    def test_http_errors_carry_status_message_and_retry_after(self):
        response = _http({"error": {"code": 429, "message": "Rate limit exceeded"}}, status=429, headers={"Retry-After": "7"})
        with self.assertRaises(OpenRouterError) as caught:
            self.chat(response)
        self.assertEqual((caught.exception.status, caught.exception.retry_after), (429, 7.0))
        self.assertIn("Rate limit exceeded", str(caught.exception))

    def test_error_inside_a_200_response_is_an_error(self):
        with self.assertRaises(OpenRouterError) as caught:
            self.chat(_http({"error": {"code": 503, "message": "provider down"}}))
        self.assertEqual(caught.exception.status, 503)
        with self.assertRaises(OpenRouterError) as caught:
            self.chat(_http({"error": {"code": "weird", "message": "?"}}))
        self.assertEqual(caught.exception.status, 502)

    def test_non_json_error_page_still_raises_with_its_status(self):
        with self.assertRaises(OpenRouterError) as caught:
            self.chat(_http(None, status=502, text="<html>Bad gateway</html>"))
        self.assertEqual(caught.exception.status, 502)
        self.assertIn("Bad gateway", str(caught.exception))

    def test_a_request_that_never_finishes_gives_up_at_the_deadline(self):
        release = threading.Event()

        def hang(*args, **kwargs):
            release.wait(5)  # stands in for a queued request that keeps the line open
            return _http({"choices": []})

        with mock.patch("openrouter.requests.post", side_effect=hang), mock.patch.object(openrouter, "REQUEST_DEADLINE_S", 0.2):
            started = time.monotonic()
            with self.assertRaises(OpenRouterError) as caught:
                OpenRouterClient(self.KEY).chat(model="m", messages=[])
            elapsed = time.monotonic() - started
        release.set()
        self.assertEqual(caught.exception.status, 408)
        self.assertLess(elapsed, 2)
        self.assertIn("too long", str(openrouter.friendly_error(caught.exception)))

    def test_errors_raised_by_requests_still_reach_the_caller(self):
        with mock.patch("openrouter.requests.post", side_effect=requests.ConnectionError("no route")):
            with self.assertRaises(requests.ConnectionError):
                OpenRouterClient(self.KEY).chat(model="m", messages=[])

    def test_the_key_never_appears_in_an_error(self):
        with self.assertRaises(OpenRouterError) as caught:
            self.chat(_http({"error": {"code": 401, "message": "No auth credentials found"}}, status=401))
        self.assertNotIn(self.KEY, str(caught.exception))
        self.assertNotIn(self.KEY, str(openrouter.friendly_error(caught.exception)))


# --- Tool declarations -------------------------------------------------------------

class ToolSchemaTests(unittest.TestCase):
    def test_every_tool_is_declared_with_json_schema_types(self):
        declared = {t["function"]["name"]: t["function"] for t in openrouter.TOOLS}
        self.assertEqual(set(declared), set(tools.TOOL_FUNCTIONS))
        allowed = {"object", "string", "number", "integer", "array", "boolean"}

        def check(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "type" and isinstance(value, str):
                        self.assertIn(value, allowed, f"{path}.type")
                    check(value, f"{path}.{key}")

        for name, function in declared.items():
            check(function["parameters"], name)
            accepted = set(inspect.signature(tools.TOOL_FUNCTIONS[name]).parameters)
            self.assertLessEqual(set(function["parameters"]["properties"]), accepted, name)
            self.assertEqual(function["parameters"]["type"], "object")

    def test_required_fields_and_nested_items_survive_the_conversion(self):
        budget = next(t["function"] for t in openrouter.TOOLS if t["function"]["name"] == "check_budget")["parameters"]
        self.assertEqual(budget["required"], ["activities", "budget"])
        self.assertEqual(budget["properties"]["activities"]["items"]["required"], ["name", "estimated_cost"])
        self.assertEqual(budget["properties"]["budget"]["type"], "number")


# --- Provider selection ------------------------------------------------------------

class ProviderTests(unittest.TestCase):
    def env(self, **values):
        """A clean environment (no real .env, no real keys) with just these variables set."""
        patches = [mock.patch("llm.load_dotenv"), mock.patch.dict(os.environ, values, clear=True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_gemini_is_the_default(self):
        self.env()
        self.assertEqual(llm.provider(), "gemini")
        self.env(GEMINI_API_KEY="g")
        self.assertEqual(llm.provider(), "gemini")

    def test_openrouter_is_chosen_when_it_is_the_only_key(self):
        self.env(OPENROUTER_API_KEY="o")
        self.assertEqual(llm.provider(), "openrouter")

    def test_with_both_keys_and_no_choice_gemini_keeps_working_as_before(self):
        self.env(OPENROUTER_API_KEY="o", GEMINI_API_KEY="g")
        self.assertEqual(llm.provider(), "gemini")

    def test_llm_provider_overrides_everything(self):
        self.env(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="o", GEMINI_API_KEY="g")
        self.assertEqual(llm.provider(), "openrouter")
        self.env(LLM_PROVIDER=" Gemini ", OPENROUTER_API_KEY="o")
        self.assertEqual(llm.provider(), "gemini")

    def test_a_typo_in_llm_provider_is_reported_not_ignored(self):
        self.env(LLM_PROVIDER="opnrouter", OPENROUTER_API_KEY="o")
        with self.assertRaisesRegex(llm.ChatError, "LLM_PROVIDER must be"):
            llm.provider()
        self.assertFalse(llm.has_api_key())
        self.assertIn("LLM_PROVIDER must be", llm.missing_key_message())
        with self.assertRaisesRegex(llm.ChatError, "LLM_PROVIDER must be"):
            llm.new_session()

    def test_has_api_key_checks_the_chosen_providers_key(self):
        self.env(LLM_PROVIDER="openrouter", GEMINI_API_KEY="g", OPENROUTER_API_KEY="  ")
        self.assertFalse(llm.has_api_key())
        self.env(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="o")
        self.assertTrue(llm.has_api_key())
        self.env(LLM_PROVIDER="gemini", OPENROUTER_API_KEY="o")
        self.assertFalse(llm.has_api_key())

    def test_missing_key_message_names_the_right_variable(self):
        self.env(LLM_PROVIDER="openrouter")
        self.assertEqual(llm.missing_key_message(), "Add your OpenRouter API key as OPENROUTER_API_KEY in the .env file, then restart the app.")
        self.env()
        self.assertEqual(llm.missing_key_message(), "Add your Gemini API key as GEMINI_API_KEY in the .env file, then restart the app.")

    def test_new_session_builds_an_openrouter_session(self):
        self.env(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="o")
        session = llm.new_session()
        self.assertIsInstance(session, OpenRouterSession)
        self.assertEqual((session.model, session.client.api_key), (openrouter.DEFAULT_MODEL, "o"))
        self.env(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="o", OPENROUTER_MODEL=" qwen/some-model:free ")
        self.assertEqual(llm.new_session().model, "qwen/some-model:free")

    def test_new_session_without_a_key_explains_what_to_do(self):
        self.env(LLM_PROVIDER="openrouter")
        with self.assertRaisesRegex(llm.ChatError, "OPENROUTER_API_KEY"):
            llm.new_session()

    def test_new_session_still_builds_a_gemini_session(self):
        self.env(LLM_PROVIDER="gemini", GEMINI_API_KEY="g", GEMINI_MODEL="some-gemini")
        with mock.patch("llm.genai.Client"):
            session = llm.new_session()
        self.assertIsInstance(session, llm.ChatSession)
        self.assertEqual(session.model, "some-gemini")


if __name__ == "__main__":
    unittest.main()
