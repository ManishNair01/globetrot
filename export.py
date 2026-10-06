"""Download formats for a finished itinerary: plain text and JSON."""
from __future__ import annotations

import re
import unicodedata

from render import TIME_ORDER, money_text, short_date
from schemas import Itinerary


def itinerary_json(it: Itinerary) -> str:
    return it.model_dump_json(indent=2)


def itinerary_text(it: Itinerary) -> str:
    diff = abs(it.budget - it.estimated_total)
    status = f"within budget, {money_text(diff, it.currency)} to spare" if it.budget_status == "within" else f"over budget by {money_text(diff, it.currency)}"
    lines = [
        f"{it.destination} - {it.total_days} day{'s' if it.total_days != 1 else ''}",
        f"Budget: {money_text(it.budget, it.currency)}",
        f"Estimated total: {money_text(it.estimated_total, it.currency)} ({status})",
    ]
    for day in sorted(it.days, key=lambda d: d.day_number):
        lines += ["", f"Day {day.day_number} - {short_date(day.date)}", day.weather_summary]
        for a in sorted(day.activities, key=lambda a: TIME_ORDER.get(a.time_of_day, 9)):
            setting = "indoor" if a.indoor else "outdoor"
            lines.append(f"  {a.time_of_day.capitalize():<10}{a.name} ({setting}) - {money_text(a.estimated_cost, it.currency)}")
    return "\n".join(lines) + "\n"


def file_stem(it: Itinerary) -> str:
    """A safe file name such as 'goa-india-itinerary' from the destination."""
    ascii_name = unicodedata.normalize("NFKD", it.destination).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.casefold()).strip("-")
    return f"{slug or 'trip'}-itinerary"
