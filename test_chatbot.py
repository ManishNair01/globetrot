"""Offline tests: tools, caching, schema reconciliation, exports, and the tool loop (with retries) on a fake Gemini client.

Run with:  python -m unittest -v
"""
from __future__ import annotations

import inspect
import unittest
from datetime import date
from unittest import mock

import requests
from google.genai import errors, types

import export
import llm
import tools
from schemas import Itinerary


class BudgetTests(unittest.TestCase):
    def test_within(self):
        r = tools.check_budget([{"name": "a", "estimated_cost": 1000}, {"name": "b", "estimated_cost": 500}], 2000)
        self.assertEqual((r["total"], r["remaining"], r["status"]), (1500, 500, "within"))

    def test_over(self):
        r = tools.check_budget([{"name": "a", "estimated_cost": 4000}, {"name": "b", "estimated_cost": 3000}], 5000)
        self.assertEqual((r["total"], r["remaining"], r["status"]), (7000, -2000, "over"))

    def test_exactly_on_budget_is_within(self):
        self.assertEqual(tools.check_budget([{"name": "a", "estimated_cost": "1,000"}], 1000)["status"], "within")

    def test_bad_input_returns_error(self):
        self.assertIn("error", tools.check_budget([{"name": "a"}], 1000))
        self.assertIn("error", tools.check_budget([], "lots"))


def _response(json_body, status=200):
    resp = mock.Mock()
    resp.status_code = status
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = json_body
    return resp


def _jaipur_responses():
    geo = _response({"results": [{"name": "Jaipur", "admin1": "Rajasthan", "country": "India", "latitude": 26.9, "longitude": 75.8}]})
    fc = _response({"daily": {
        "time": ["2026-10-03", "2026-10-04"], "temperature_2m_max": [33, 34], "temperature_2m_min": [22, 23],
        "precipitation_probability_max": [10, 80], "weather_code": [1, 63],
    }})
    return geo, fc


class WeatherTests(unittest.TestCase):
    TODAY = date(2026, 10, 1)

    def setUp(self):
        tools.clear_cache()

    def test_success(self):
        geo = _response({"results": [{"name": "Jaipur", "admin1": "Rajasthan", "country": "India", "latitude": 26.9, "longitude": 75.8}]})
        fc = _response({"daily": {
            "time": ["2026-10-03", "2026-10-04"], "temperature_2m_max": [33, 34], "temperature_2m_min": [22, 23],
            "precipitation_probability_max": [10, 80], "weather_code": [1, 63],
        }})
        with mock.patch("tools.requests.get", side_effect=[geo, fc]):
            r = tools.get_weather("Jaipur", "2026-10-03", 2, today=self.TODAY)
        self.assertEqual(r["place"], "Jaipur, Rajasthan, India")
        self.assertEqual([d["conditions"] for d in r["forecast"]], ["Mostly clear", "Rain"])

    def test_fuzzy_first_hit_is_skipped_for_exact_name_match(self):
        hits = [
            {"name": "Genoa", "country": "Italy", "population": 580097, "latitude": 1, "longitude": 1},
            {"name": "Calangute", "admin1": "Goa", "country": "India", "population": 17446, "latitude": 2, "longitude": 2},
            {"name": "Calangute", "admin1": "Other", "country": "India", "population": 5, "latitude": 3, "longitude": 3},
        ]
        picked = tools._pick_place(hits, "calangute")
        self.assertEqual(picked["admin1"], "Goa")
        self.assertEqual(tools._pick_place(hits[:1], "Goa")["name"], "Genoa")  # no exact match: keep API order

    def test_country_code_is_sent_to_geocoder(self):
        geo = _response({"results": [{"name": "Jaipur", "country": "India", "latitude": 1, "longitude": 1}]})
        fc = _response({"daily": {"time": ["2026-10-03"]}})
        with mock.patch("tools.requests.get", side_effect=[geo, fc]) as get:
            tools.get_weather("Jaipur", "2026-10-03", 1, country_code="in", today=self.TODAY)
        self.assertEqual(get.call_args_list[0].kwargs["params"]["countryCode"], "IN")

    def test_unknown_city(self):
        with mock.patch("tools.requests.get", return_value=_response({})):
            self.assertIn("Could not find", tools.get_weather("Zzzzz", "2026-10-03", 2, today=self.TODAY)["error"])

    def test_date_beyond_forecast(self):
        with mock.patch("tools.requests.get") as get:
            r = tools.get_weather("Goa", "2026-11-15", 3, today=self.TODAY)
        get.assert_not_called()
        self.assertIn("No forecast available", r["error"])

    def test_past_and_bad_dates(self):
        self.assertIn("past", tools.get_weather("Goa", "2026-09-01", 3, today=self.TODAY)["error"])
        self.assertIn("YYYY-MM-DD", tools.get_weather("Goa", "next friday", 3, today=self.TODAY)["error"])

    def test_network_error_does_not_raise(self):
        with mock.patch("tools.requests.get", side_effect=requests.ConnectionError("offline")):
            self.assertIn("unreachable", tools.get_weather("Goa", "2026-10-03", 2, today=self.TODAY)["error"])

    def test_run_tool_unknown_and_bad_args(self):
        self.assertIn("Unknown tool", tools.run_tool("nope", {})["error"])
        self.assertIn("Bad arguments", tools.run_tool("check_budget", {"wrong": 1})["error"])

    def test_repeat_lookup_is_served_from_cache(self):
        with mock.patch("tools.requests.get", side_effect=_jaipur_responses()) as get:
            first = tools.get_weather("Jaipur", "2026-10-03", 2, today=self.TODAY)
            second = tools.get_weather(" jaipur ", "2026-10-03", 2, today=self.TODAY)
        self.assertEqual(get.call_count, 2)  # one geocode + one forecast, not four calls
        self.assertEqual(first, second)

    def test_failures_are_not_cached(self):
        with mock.patch("tools.requests.get", side_effect=requests.ConnectionError("offline")):
            self.assertIn("unreachable", tools.get_weather("Jaipur", "2026-10-03", 2, today=self.TODAY)["error"])
        with mock.patch("tools.requests.get", side_effect=_jaipur_responses()):
            self.assertNotIn("error", tools.get_weather("Jaipur", "2026-10-03", 2, today=self.TODAY))

    def test_cache_entries_expire(self):
        with mock.patch("tools.time.monotonic", return_value=1000.0), mock.patch("tools.requests.get", side_effect=_jaipur_responses()):
            tools.get_weather("Jaipur", "2026-10-03", 2, today=self.TODAY)
        later = 1000.0 + tools.CACHE_TTL_S + 1
        with mock.patch("tools.time.monotonic", return_value=later), mock.patch("tools.requests.get", side_effect=_jaipur_responses()) as get:
            tools.get_weather("Jaipur", "2026-10-03", 2, today=self.TODAY)
        self.assertEqual(get.call_count, 2)


class CurrencyTests(unittest.TestCase):
    def setUp(self):
        tools.clear_cache()

    def test_converts_at_the_reference_rate(self):
        quote = _response({"amount": 1.0, "base": "EUR", "date": "2026-09-30", "rates": {"INR": 108.8205}})
        with mock.patch("tools.requests.get", return_value=quote) as get:
            r = tools.convert_currency(100, "eur", "inr")
        self.assertEqual((r["from"], r["to"], r["rate"], r["converted"], r["rate_date"]), ("EUR", "INR", 108.8205, 10882.05, "2026-09-30"))
        self.assertEqual(get.call_args.kwargs["params"], {"base": "EUR", "symbols": "INR"})

    def test_same_currency_needs_no_lookup(self):
        with mock.patch("tools.requests.get") as get:
            r = tools.convert_currency(500, "INR", "inr")
        get.assert_not_called()
        self.assertEqual((r["rate"], r["converted"]), (1.0, 500.0))

    def test_bad_input_returns_error(self):
        self.assertIn("3-letter", tools.convert_currency(1, "rupees", "EUR")["error"])
        self.assertIn("3-letter", tools.convert_currency(1, "EUR", "")["error"])
        self.assertIn("amount", tools.convert_currency("lots", "EUR", "INR")["error"])
        self.assertIn("amount", tools.convert_currency(-5, "EUR", "INR")["error"])

    def test_unsupported_currency(self):
        with mock.patch("tools.requests.get", return_value=_response({"message": "not found"}, status=404)):
            self.assertIn("not available", tools.convert_currency(1, "EUR", "ZZZ")["error"])

    def test_network_error_does_not_raise(self):
        with mock.patch("tools.requests.get", side_effect=requests.ConnectionError("offline")):
            self.assertIn("unreachable", tools.convert_currency(1, "EUR", "INR")["error"])

    def test_unexpected_response_shape(self):
        with mock.patch("tools.requests.get", return_value=_response({"date": "2026-09-30", "rates": {}})):
            self.assertIn("unexpected format", tools.convert_currency(1, "EUR", "INR")["error"])

    def test_rate_is_cached(self):
        quote = _response({"date": "2026-09-30", "rates": {"INR": 100.0}})
        with mock.patch("tools.requests.get", return_value=quote) as get:
            tools.convert_currency(1, "EUR", "INR")
            r = tools.convert_currency(3, "EUR", "INR")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(r["converted"], 300.0)

    def test_run_tool_dispatches_it(self):
        quote = _response({"date": "2026-09-30", "rates": {"INR": 100.0}})
        with mock.patch("tools.requests.get", return_value=quote):
            r = tools.run_tool("convert_currency", {"amount": 2, "from_currency": "EUR", "to_currency": "INR"})
        self.assertEqual(r["converted"], 200.0)


# --- Fake Gemini client ------------------------------------------------------

def _call(name, **args):
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])


def _text(text):
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def _resp(content=None, text=None):
    r = mock.Mock()
    r.candidates = [mock.Mock(content=content)] if content is not None else []
    r.text = text
    return r


def _api_error(code, retry_delay=None):
    body = {"error": {"message": "x"}}
    if retry_delay:
        body["error"]["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return errors.APIError(code, body)


class FakeClient:
    """Returns queued responses in order and records every request."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        # Snapshot: the session keeps appending to the same list after the call returns.
        self.requests.append((model, list(contents) if isinstance(contents, list) else contents, config))
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


ITINERARY_JSON = """{"destination": "Goa, India", "total_days": 1, "currency": "INR", "budget": 1000,
 "days": [{"day_number": 1, "date": "2026-10-03", "weather_summary": "Sunny, 31C",
   "activities": [{"name": "Beach", "time_of_day": "morning", "indoor": false, "estimated_cost": 700},
                  {"name": "Museum", "time_of_day": "afternoon", "indoor": true, "estimated_cost": 600}]}],
 "estimated_total": 1, "budget_status": "within"}"""


class SessionTests(unittest.TestCase):
    def test_planning_turn_runs_tools_then_formats(self):
        client = FakeClient(
            _resp(_call("get_weather", city="Goa", start_date="2026-10-03", days=1)),
            _resp(_call("check_budget", activities=[{"name": "Beach", "estimated_cost": 700}, {"name": "Museum", "estimated_cost": 600}], budget=1000)),
            _resp(_text("Day 1: beach, museum.")),
            _resp(text=ITINERARY_JSON),
        )
        events = []
        with mock.patch("llm.run_tool", side_effect=lambda n, a: tools.check_budget(**a) if n == "check_budget" else {"place": "Goa, India", "forecast": [{}]}):
            result = llm.ChatSession(client, "m").send("1 day in Goa, 1000", lambda k, m: events.append((k, m)))

        self.assertEqual([c.name for c in result.tool_calls], ["get_weather", "check_budget"])
        self.assertIsNotNone(result.itinerary)
        # The model claimed total 1 / within; Python recomputes 1300 / over.
        self.assertEqual((result.itinerary.estimated_total, result.itinerary.budget_status), (1300, "over"))
        self.assertIn("Checking weather for Goa…", [m for _, m in events])
        self.assertEqual([k for k, _ in events][-1], "format")

    def test_question_turn_makes_no_tool_call_and_no_itinerary(self):
        client = FakeClient(_resp(_text("Where would you like to go, and for how long?")))
        result = llm.ChatSession(client, "m").send("plan me a trip")
        self.assertIsNone(result.itinerary)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(len(client.requests), 1)

    def test_bad_json_retries_once_then_falls_back_to_text(self):
        client = FakeClient(
            _resp(_call("check_budget", activities=[{"name": "a", "estimated_cost": 1}], budget=10)),
            _resp(_text("Draft plan")),
            _resp(text="not json"), _resp(text="{}"),
        )
        result = llm.ChatSession(client, "m").send("go")
        self.assertIsNone(result.itinerary)
        self.assertEqual(result.text, "Draft plan")
        self.assertEqual(len(client.requests), 4)  # 2 loop calls + 2 format attempts

    def test_tool_loop_is_capped(self):
        looping = [_resp(_call("get_weather", city="Goa", start_date="2026-10-03", days=1)) for _ in range(llm.MAX_TOOL_ROUNDS)]
        session = llm.ChatSession(FakeClient(*looping), "m")
        with mock.patch("llm.run_tool", return_value={"forecast": []}):
            with self.assertRaises(llm.ChatError):
                session.send("go")
        self.assertEqual(session.contents, [])  # failed turn leaves no history behind

    def test_api_errors_become_friendly_messages(self):
        for code, message in [(401, "rejected the API key"), (404, "could not find the model"), (500, "temporarily unavailable"), (400, "Try rephrasing")]:
            session = llm.ChatSession(FakeClient(_api_error(code)), "m")
            with self.assertRaisesRegex(llm.ChatError, message):
                session.send("hi")
            self.assertEqual(session.contents, [])

    def test_rate_limit_is_retried_then_succeeds(self):
        client = FakeClient(_api_error(429), _api_error(503), _resp(_text("Where to?")))
        events = []
        with mock.patch("llm.time.sleep") as sleep:
            result = llm.ChatSession(client, "m").send("hi", lambda k, m: events.append((k, m)))
        self.assertEqual(result.text, "Where to?")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 6])
        self.assertEqual(events, [("retry", "Gemini is busy. Retrying in 2s…"), ("retry", "Gemini is busy. Retrying in 6s…")])

    def test_rate_limit_gives_up_after_the_last_retry(self):
        client = FakeClient(_api_error(429), _api_error(429), _api_error(429))
        session = llm.ChatSession(client, "m")
        with mock.patch("llm.time.sleep") as sleep, self.assertRaisesRegex(llm.ChatError, "rate-limiting"):
            session.send("hi")
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(client.queue, [])  # all three attempts were used
        self.assertEqual(session.contents, [])

    def test_errors_that_retrying_cannot_fix_are_not_retried(self):
        client = FakeClient(_api_error(401), _resp(_text("never reached")))
        with mock.patch("llm.time.sleep") as sleep, self.assertRaisesRegex(llm.ChatError, "rejected the API key"):
            llm.ChatSession(client, "m").send("hi")
        sleep.assert_not_called()
        self.assertEqual(len(client.queue), 1)

    def test_wait_requested_by_gemini_is_honoured(self):
        client = FakeClient(_api_error(429, retry_delay="7s"), _resp(_text("ok")))
        with mock.patch("llm.time.sleep") as sleep:
            llm.ChatSession(client, "m").send("hi")
        sleep.assert_called_once_with(8)  # the 7s Gemini asked for, plus a second of margin

    def test_long_wait_requested_by_gemini_fails_fast(self):
        client = FakeClient(_api_error(429, retry_delay="34s"), _resp(_text("never reached")))
        with mock.patch("llm.time.sleep") as sleep, self.assertRaisesRegex(llm.ChatError, "rate-limiting"):
            llm.ChatSession(client, "m").send("hi")
        sleep.assert_not_called()

    def test_formatting_call_is_retried_too(self):
        client = FakeClient(
            _resp(_call("check_budget", activities=[{"name": "Beach", "estimated_cost": 700}], budget=1000)),
            _resp(_text("Draft plan")),
            _api_error(429),
            _resp(text=ITINERARY_JSON),
        )
        with mock.patch("llm.time.sleep"):
            result = llm.ChatSession(client, "m").send("go")
        self.assertIsNotNone(result.itinerary)

    def test_every_declared_tool_has_a_matching_function(self):
        declared = {d.name: d for t in llm.TOOLS for d in t.function_declarations}
        self.assertEqual(set(declared), set(tools.TOOL_FUNCTIONS))
        for name, declaration in declared.items():
            accepted = set(inspect.signature(tools.TOOL_FUNCTIONS[name]).parameters)
            self.assertLessEqual(set(declaration.parameters.properties), accepted, name)

    def test_currency_status_lines(self):
        self.assertEqual(llm._describe_call("convert_currency", {"from_currency": "EUR", "to_currency": "INR"}), "Converting EUR to INR…")
        result = {"from": "EUR", "to": "INR", "rate": 108.8205, "rate_date": "2026-09-30"}
        self.assertEqual(llm._describe_result("convert_currency", result), "1 EUR = 108.8 INR · 2026-09-30")
        self.assertTrue(llm._describe_result("convert_currency", {"error": "down"}).startswith("Note:"))

    def test_offline_becomes_friendly_message(self):
        import httpx
        session = llm.ChatSession(FakeClient(httpx.ConnectError("no route")), "m")
        with self.assertRaisesRegex(llm.ChatError, "internet"):
            session.send("hi")

    def test_history_carries_into_follow_up(self):
        client = FakeClient(_resp(_text("Sure.")), _resp(_text("Done.")))
        session = llm.ChatSession(client, "m")
        session.send("first")
        session.send("second")
        self.assertEqual(len(client.requests[1][1]), 3)  # user, model, user


class ExportTests(unittest.TestCase):
    # The model claimed 1 / within; Python recomputes 1300 against a 1000 budget.
    ITINERARY = llm._reconcile(Itinerary.model_validate_json(ITINERARY_JSON))

    def test_text_download(self):
        text = export.itinerary_text(self.ITINERARY)
        self.assertEqual(text.splitlines()[:3], [
            "Goa, India - 1 day",
            "Budget: ₹1,000",
            "Estimated total: ₹1,300 (over budget by ₹300)",
        ])
        self.assertIn("Day 1 - Sat 03 Oct\nSunny, 31C", text)
        self.assertIn("  Morning   Beach (outdoor) - ₹700", text)
        self.assertIn("  Afternoon Museum (indoor) - ₹600", text)
        self.assertLess(text.index("Beach"), text.index("Museum"))
        self.assertTrue(text.endswith("\n"))

    def test_json_download_round_trips(self):
        self.assertEqual(Itinerary.model_validate_json(export.itinerary_json(self.ITINERARY)), self.ITINERARY)

    def test_file_stem_is_filesystem_safe(self):
        stem = lambda destination: export.file_stem(self.ITINERARY.model_copy(update={"destination": destination}))
        self.assertEqual(stem("Goa, India"), "goa-india-itinerary")
        self.assertEqual(stem("Zürich"), "zurich-itinerary")
        self.assertEqual(stem("../../etc/passwd"), "etc-passwd-itinerary")
        self.assertEqual(stem("東京"), "trip-itinerary")


if __name__ == "__main__":
    unittest.main()
