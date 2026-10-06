"""Live smoke test: one real call to each outside service. Not part of the offline test suite.

    .venv/Scripts/python.exe smoke.py

Checks Open-Meteo and Frankfurter (no key needed), then one short chat turn per provider whose key is in
.env. A provider without a key is SKIPPED, not failed. Exit code is 1 if anything that ran failed.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from typing import Callable, List, Tuple

from dotenv import load_dotenv

import llm
import openrouter
import tools

load_dotenv()


def check_weather() -> str:
    result = tools.get_weather("Jaipur", (date.today() + timedelta(days=1)).isoformat(), 2)
    if "error" in result:
        raise RuntimeError(result["error"])
    return f"{result['place']}, {len(result['forecast'])} forecast days"


def check_currency() -> str:
    result = tools.convert_currency(100, "EUR", "INR")
    if "error" in result:
        raise RuntimeError(result["error"])
    return f"100 EUR = {result['converted']} INR"


def _chat(make: Callable[[], object]) -> str:
    session = make()
    result = session.send("Reply with the single word: ready")  # type: ignore[attr-defined]
    if not result.text.strip():
        raise RuntimeError("empty reply")
    return f"{len(result.text)} chars: {result.text.strip()[:40]!r}"


def check_gemini() -> str:
    return _chat(lambda: llm.ChatSession(*llm.make_client()))


def check_openrouter() -> str:
    return _chat(openrouter.new_session)


def main() -> int:
    checks: List[Tuple[str, Callable[[], str], str]] = [
        ("Open-Meteo weather", check_weather, ""),
        ("Frankfurter rates", check_currency, ""),
        ("Gemini chat", check_gemini, "GEMINI_API_KEY"),
        ("OpenRouter chat", check_openrouter, "OPENROUTER_API_KEY"),
    ]
    failed = False
    for name, check, key_var in checks:
        if key_var and not os.getenv(key_var, "").strip():
            print(f"SKIP  {name} ({key_var} not set)")
            continue
        started = time.monotonic()
        try:
            detail = check()
        except Exception as exc:  # report every failure, keep going
            failed = True
            print(f"FAIL  {name}: {exc} ({time.monotonic() - started:.1f}s)")
        else:
            print(f"ok    {name}: {detail} ({time.monotonic() - started:.1f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
