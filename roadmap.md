> **Outdated.** This plan predates the OpenRouter provider, currency tool, form page, site and HTTP API. The current picture is in [backend-skeleton.md](backend-skeleton.md) and [frontend-skeleton.md](frontend-skeleton.md); the code wins over this file.

# Travel Chatbot — Roadmap

Milestones in order. Tick them off as you go. Dates are left blank because the class deadline isn't set yet.

## Milestone 1 — Foundations
- [ ] Environment and dependencies installed
- [ ] Gemini API key working
- [ ] Weather tool works from the command line
- [ ] Budget tool tested

**Done when:** both tools return correct results outside any chatbot.

## Milestone 2 — Working chatbot
- [ ] Function-calling loop working
- [ ] Bot asks for missing details
- [ ] Bot calls the weather tool before planning
- [ ] Plan mentions real forecast values

**Done when:** a terminal conversation produces a weather-aware plan.

## Milestone 3 — Demo-ready app
- [ ] Structured JSON itinerary validated by Pydantic
- [ ] Streamlit chat UI with per-day tables
- [ ] Tool-call status indicator
- [ ] Budget banner
- [ ] Error handling for API failures

**Done when:** all six acceptance criteria in `requirements.md` §9 pass.

## Milestone 4 — Presentation
- [ ] Three demo queries scripted and rehearsed
- [ ] Fallback screen recording saved
- [ ] One flow slide prepared
- [ ] Full dry run on the machine you'll present from

**Done when:** you have run the demo twice without touching the code.

## Later (only if time allows)
- [ ] Follow-up edits ("make day 2 cheaper")
- [ ] Download itinerary as text or JSON
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add a second tool (e.g. currency conversion via a free API)

## Explicitly not planned
RAG, user accounts, bookings, live prices, maps.
