> **Outdated.** This plan predates the OpenRouter provider, currency tool, form page, site and HTTP API. The current picture is in [backend-skeleton.md](backend-skeleton.md) and [frontend-skeleton.md](frontend-skeleton.md); the code wins over this file.

# Travel Chatbot — Build Plan

How to build it, in order. Each step ends with something you can run and check. Steps are sized in hours so you can compress or stretch them to fit your time.

## Architecture

```
User ──> Streamlit chat UI
              │
              ▼
        Gemini (with tools)
         │            ▲
   tool call     tool result
         ▼            │
   get_weather()  check_budget()
         │
         ▼
   Open-Meteo API

Final step: Gemini formats the answer as JSON (itinerary schema)
            ──> Streamlit renders tables
```

Two model calls per turn, on purpose: first a tool-calling loop to gather facts, then a separate formatting call that returns the JSON schema. Combining tools and strict JSON output in one call is not reliably supported across models, so splitting them avoids that problem.

## Project layout

```
chatbot/
  app.py            # Streamlit UI and chat loop
  llm.py            # Gemini client, tool declarations, tool-calling loop
  tools.py          # get_weather(), check_budget()
  schemas.py        # Pydantic models for the itinerary
  prompts.py        # system prompt and formatting prompt
  .env              # GEMINI_API_KEY, GEMINI_MODEL (not committed)
  requirements.txt
  requirements.md
  build-plan.md
  roadmap.md
```

## Steps

### Step 1 — Setup (0.5 h)
- Create a virtual environment, install the dependencies.
- Get a Gemini API key, put it in `.env`. Add `.env` to `.gitignore`.
- **Check:** a 5-line script prints a Gemini reply.

### Step 2 — Weather tool, standalone (1–1.5 h)
- In `tools.py`, write `get_weather(city, start_date, days)`:
  1. Geocode the city with Open-Meteo geocoding.
  2. Fetch the daily forecast (max/min temperature, precipitation probability, weather code).
  3. Return a small list of dicts, one per day, plus the matched place name.
- Handle: city not found, dates beyond the forecast range, network error (return an error message the model can read, don't raise).
- **Check:** run it from the command line for 3 cities, including a bad name and a date 30 days out.

### Step 3 — Budget tool (0.5 h)
- `check_budget(activities, budget)` adds up the estimated costs and returns total, remaining, and `within`/`over`.
- **Check:** unit-test it with 3 hand-written inputs.

### Step 4 — Function calling loop (2 h)
- In `llm.py`, declare both tools for Gemini.
- Write the loop: send the message → if the model asks for a tool, run it, send back the result → repeat until the model returns text (cap at ~5 iterations).
- System prompt: ask for missing details before planning, always call the weather tool before planning, never invent weather.
- **Check:** in a terminal, "3 days in Jaipur from Saturday" triggers the weather call and the reply mentions the actual forecast.

### Step 5 — Structured output (1–1.5 h)
- Define the schema in `schemas.py` (see requirements §8).
- Second call: pass the gathered facts and ask for the itinerary as JSON matching the schema; validate with Pydantic.
- On validation failure, retry once, then show a plain-text fallback.
- **Check:** the output parses into the Pydantic model for 5 different requests.

### Step 6 — Streamlit UI (1.5–2 h)
- Chat interface with `st.chat_message` and `st.chat_input`; keep history in `st.session_state`.
- Render each day as a table (time of day, activity, indoor/outdoor, cost) with the weather summary as a caption.
- Show a status line while tools run (`st.status` or `st.spinner`).
- Show the budget status as a coloured banner.
- **Check:** the full flow works in the browser.

### Step 7 — Error handling and polish (1 h)
- Friendly messages for API failures and rate limits.
- Reset chat button.
- Test the edge cases from the acceptance criteria.

### Step 8 — Demo prep (1–1.5 h)
- Script three queries:
  1. **Normal:** "4 days in Goa, ₹20,000, starting this Friday."
  2. **Tool showcase:** a destination with rain in the forecast, to show the plan change.
  3. **Edge case:** "Paris for a week, ₹5,000" to show the budget warning.
- Record a short screen capture as a fallback in case wifi or the API fails.
- One slide on the flow: user → model → tool call → result → itinerary.

## Total estimate

About 9–12 hours of focused work. If you have less time, do steps 1–4 and 6 with plain-text output, and skip step 5.

## Risks

| Risk | Mitigation |
|------|------------|
| Gemini free-tier rate limit during the demo | Keep requests few, use the recording as backup |
| Trip dates beyond the 16-day forecast | Tool returns a clear "no forecast" message; the prompt tells the model to say so |
| Model skips the weather tool | Explicit system prompt; log tool calls to check |
| JSON output fails validation | One retry, then a text fallback |
| City name is ambiguous | Show the matched place name to the user |
