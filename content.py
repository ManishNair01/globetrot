"""Read access to the site's content store (super-travel/journeys.json) for the concierge and the API.

The file is validated and turned into pages by super-travel/build.py; nothing here writes it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

JOURNEYS_FILE = Path(__file__).parent / "super-travel" / "journeys.json"
_SLUG = re.compile(r"[a-z0-9-]{1,40}")


def load_journeys() -> Dict[str, Any]:
    """The whole store: {currency, journeys[]}. Read on every call so edits show without a restart."""
    return json.loads(JOURNEYS_FILE.read_text(encoding="utf-8"))


def find_journey(slug: Any) -> Optional[Dict[str, Any]]:
    """The journey with this slug, or None for an unknown or malformed slug."""
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug):
        return None
    return next((j for j in load_journeys()["journeys"] if j["slug"] == slug), None)


def journey_prompt(slug: Any) -> Optional[str]:
    """The opening message for 'plan a trip like this journey', or None if the slug is unknown."""
    journey = find_journey(slug)
    if journey is None:
        return None
    currency = load_journeys().get("currency", "")
    text = f"I would like a trip like your journey “{journey['title']}” ({journey['slug'].title()}): {journey['days']} days"
    if journey.get("budget"):
        text += f", about {currency}{int(journey['budget']):,}"
    return text + ". Ask me for anything you still need."
