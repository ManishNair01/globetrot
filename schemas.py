"""Itinerary schema (requirements section 8).

`Itinerary.budget` is an addition to that schema: the budget banner needs the
user's limit, and `estimated_total`/`budget_status` are recomputed from it.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel


class Activity(BaseModel):
    name: str
    time_of_day: Literal["morning", "afternoon", "evening"]
    indoor: bool
    estimated_cost: float


class Day(BaseModel):
    day_number: int
    date: str
    weather_summary: str
    activities: List[Activity]


class Itinerary(BaseModel):
    destination: str
    total_days: int
    currency: str
    budget: float
    days: List[Day]
    estimated_total: float
    budget_status: Literal["within", "over"]
