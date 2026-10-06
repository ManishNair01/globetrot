"""HTML for the GlobeTrot look: nav, hero, itinerary tables, budget banner, notices.

Everything the model or the user wrote is HTML-escaped. Markup is emitted without
newlines or indentation so Streamlit's markdown pass leaves it alone.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Iterable

from schemas import Itinerary

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "AED": "AED "}
TIME_ORDER = {"morning": 0, "afternoon": 1, "evening": 2}


def esc(text: object) -> str:
    # "$" would be read as LaTeX by Streamlit's markdown, so emit it as an entity.
    return escape(str(text)).replace("$", "&#36;")


def money_text(amount: float, currency: str) -> str:
    code = (currency or "").strip()
    symbol = CURRENCY_SYMBOLS.get(code.upper(), code if len(code) == 1 else f"{code} " if code else "")
    sign = "-" if amount < 0 else ""
    return f"{sign}{symbol}{abs(amount):,.0f}"


def money(amount: float, currency: str) -> str:
    return esc(money_text(amount, currency))


def short_date(iso: str) -> str:
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%a %d %b")
    except ValueError:
        return iso


def nav_html(site_url: str) -> str:
    base = esc(site_url.rstrip("/"))
    return (
        '<header class="st-nav"><div class="st-nav__inner">'
        f'<a class="st-nav__brand" href="{base}/" target="_self">GlobeTrot</a>'
        '<nav class="st-nav__menu">'
        f'<a href="{base}/#services" target="_self">Services</a>'
        f'<a href="{base}/journeys/" target="_self">Journeys</a>'
        '<span class="is-current">Concierge</span>'
        '</nav></div></header>'
    )


def hero_html() -> str:
    return (
        '<section class="st-hero">'
        '<p class="label">Concierge</p>'
        '<h1 class="display">Plan your<br><em>journey.</em></h1>'
        '<p class="st-hero__lede">Tell us where, for how long, and what you would like to spend. '
        'We check the forecast, plan each day around it, and keep the total within your budget.</p>'
        '</section>'
    )


def notice_html(message: str) -> str:
    return f'<div class="notice">{esc(message)}</div>'


def user_html(text: str) -> str:
    return f'<p class="user-text">{esc(text)}</p>'


def banner_html(it: Itinerary) -> str:
    diff = abs(it.budget - it.estimated_total)
    if it.budget_status == "within":
        body = f"Within budget &middot; {money(it.estimated_total, it.currency)} of {money(it.budget, it.currency)} &middot; {money(diff, it.currency)} to spare"
    else:
        body = f"Over budget &middot; {money(it.estimated_total, it.currency)} against {money(it.budget, it.currency)} &middot; {money(diff, it.currency)} over"
    return f'<div class="banner banner--{it.budget_status}">{body}</div>'


def _rows(day, currency: str) -> str:
    activities = sorted(day.activities, key=lambda a: TIME_ORDER.get(a.time_of_day, 9))
    return "".join(
        '<tr>'
        f'<td class="t-when">{esc(a.time_of_day)}</td>'
        f'<td class="t-what">{esc(a.name)}</td>'
        f'<td class="t-where"><span class="pill pill--{"in" if a.indoor else "out"}">{"Indoor" if a.indoor else "Outdoor"}</span></td>'
        f'<td class="t-cost">{money(a.estimated_cost, currency)}</td>'
        '</tr>'
        for a in activities
    )


def _day(day, currency: str) -> str:
    return (
        '<article class="day">'
        f'<div class="day__top"><span class="day__num">Day {day.day_number:02d}</span>'
        f'<span class="day__date">{esc(short_date(day.date))}</span></div>'
        f'<p class="day__weather">{esc(day.weather_summary)}</p>'
        '<table class="day__table"><thead><tr><th>When</th><th>Activity</th><th>Setting</th><th>Cost</th></tr></thead>'
        f'<tbody>{_rows(day, currency)}</tbody></table>'
        '</article>'
    )


def itinerary_html(it: Itinerary) -> str:
    days: Iterable = sorted(it.days, key=lambda d: d.day_number)
    return (
        '<section class="itin">'
        '<header class="itin__head"><p class="label">Your itinerary</p>'
        f'<h3 class="display">{esc(it.destination)}</h3>'
        f'<p class="itin__sub">{it.total_days} day{"s" if it.total_days != 1 else ""} &middot; {money(it.budget, it.currency)} budget</p></header>'
        f'{banner_html(it)}'
        f'{"".join(_day(d, it.currency) for d in days)}'
        f'<div class="itin__total"><span>Estimated total</span><strong>{money(it.estimated_total, it.currency)}</strong></div>'
        '</section>'
    )
