> **Outdated.** This plan predates the OpenRouter provider, currency tool, form page, site and HTTP API. The current picture is in [backend-skeleton.md](backend-skeleton.md) and [frontend-skeleton.md](frontend-skeleton.md); the code wins over this file.

# Travel Chatbot — Requirements

Class-session demo. Not a production product. No RAG.

## 1. Goal

A chatbot that collects a trip request (destination, days, budget), checks the live weather, and returns a day-by-day itinerary that adapts to the forecast and is checked against the budget.

The point of the demo: show that a chatbot with **tools** (function calling) does more than a plain prompt.

## 2. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | Chat UI where the user describes a trip in natural language | Must |
| F2 | Bot asks follow-up questions if destination, number of days, start date, or budget is missing | Must |
| F3 | Bot calls a **weather tool** (Open-Meteo) for the destination and trip dates | Must |
| F4 | Itinerary adapts to the forecast (e.g. indoor activities on rainy days) | Must |
| F5 | Itinerary is returned as structured JSON and rendered as a table per day | Must |
| F6 | **Budget check** function totals estimated costs and flags when over budget | Should |
| F7 | Visible indicator when a tool is called (e.g. "Checking weather for Goa...") | Should |
| F8 | Conversation history kept within the session so the user can ask for changes ("make day 2 cheaper") | Should |
| F9 | "Reset chat" button | Could |
| F10 | Download itinerary as text or JSON | Could |

## 3. Non-functional requirements

- **Cost:** free tiers only (Gemini API free tier, Open-Meteo has no key).
- **Setup:** runs locally with `pip install -r requirements.txt` and `streamlit run app.py`.
- **Resilience:** if the weather API or Gemini fails, show a readable message and do not crash.
- **Secrets:** API key is read from a `.env` file and never committed.
- **Response time:** a normal request should finish in under ~15 seconds.

## 4. Out of scope

- RAG or any document retrieval
- User accounts, login, payments, or a database
- Booking flights or hotels, or live prices
- Maps
- Deployment beyond running locally (optional: Streamlit Community Cloud)

## 5. Tech stack

| Layer | Choice | Reason |
|-------|--------|--------|
| UI | Streamlit (`st.chat_message`, `st.chat_input`) | Fastest chat UI in Python |
| LLM | Gemini API via the `google-genai` SDK | Free tier, supports function calling and JSON schema output |
| Weather | Open-Meteo geocoding + forecast APIs | Free, no API key |
| HTTP | `requests` | Simple |
| Config | `python-dotenv` | Keeps the key out of code |
| Validation | `pydantic` | Defines the itinerary schema |

Python 3.10+.

## 6. Python dependencies (`requirements.txt`)

```
streamlit
google-genai
requests
python-dotenv
pydantic
```

## 7. External services and constraints

- **Gemini API key** from Google AI Studio. Free tier has rate limits, so avoid rapid repeated calls during the demo. Check the current model names in the docs; keep the model name in `.env` so it can be swapped.
- **Open-Meteo forecast** covers roughly the next 16 days. Trips starting further out cannot be forecast, so the bot must say so and fall back to typical seasonal advice instead of inventing weather.
- **Open-Meteo geocoding** can return several places with the same name. Use the top result and show which place was matched.

## 8. Data model (itinerary schema)

```
Itinerary
  destination: str
  total_days: int
  currency: str
  days: list[Day]
  estimated_total: float
  budget_status: "within" | "over"

Day
  day_number: int
  date: str
  weather_summary: str
  activities: list[Activity]

Activity
  name: str
  time_of_day: "morning" | "afternoon" | "evening"
  indoor: bool
  estimated_cost: float
```

## 9. Acceptance criteria

1. Typing "3 days in Jaipur, ₹15,000, starting Saturday" returns a table-rendered itinerary with weather per day.
2. A rainy forecast produces more indoor activities that day than a sunny one.
3. A budget of ₹5,000 for a long international trip is flagged as over budget or unrealistic, not silently accepted.
4. A request with no destination gets a follow-up question, not an itinerary.
5. Turning off the internet produces an error message, not a stack trace.
6. The three scripted demo queries (see the build plan) work end to end.

## 10. Open questions

- Class time limit for the demo?
- Any required technique or technology the session must show?
- Currency: INR by default, unless the class wants otherwise.
