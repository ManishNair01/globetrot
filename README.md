# GlobeTrot

A private travel concierge. Tell it where, for how long and what you would like to spend, and it plans a day-by-day itinerary that follows the live weather forecast and is checked against your budget.

This is a class project with three parts:

| Part | What it is | Port |
|---|---|---|
| **Site** (`super-travel/`) | Static marketing site: home page, journeys list, six journey pages | 5173 |
| **Concierge** (repo root) | Streamlit chat app. Describe a trip in a message, or fill in a form | 8501 |
| **API** (`api.py`) | Optional FastAPI layer over the same concierge, for pages that cannot use Streamlit | 8000 |

## How the concierge works

Each turn makes two model calls:

1. **Tool loop.** The model calls tools to gather facts, then drafts a plan.
   - `get_weather`: forecast from Open-Meteo (no key needed)
   - `check_budget`: totals the costs and flags over-budget plans
   - `convert_currency`: rates from Frankfurter (no key needed)
2. **Structured itinerary.** The draft is turned into JSON, validated with Pydantic, and shown as a table per day. Totals are recomputed in Python, never trusted from the model.

It works with Gemini or OpenRouter, chosen in `.env`.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux
cp .env.example .env
```

Then edit `.env`:

- `LLM_PROVIDER`: `gemini` or `openrouter`
- `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey), or `OPENROUTER_API_KEY` from [openrouter.ai/keys](https://openrouter.ai/keys)
- `SITE_URL`: where the site runs (default `http://localhost:5173`)

`.env` is git-ignored. Never commit it. Restart the app after changing it, because it is read at startup.

## Run

```bash
.venv/Scripts/python.exe -m streamlit run app.py --server.port 8501   # concierge
python super-travel/serve.py 5173                                     # site
.venv/Scripts/python.exe -m uvicorn api:app --port 8000               # API (optional)
```

Open http://localhost:5173 for the site and http://localhost:8501 for the concierge. Try: *"4 days in Goa, ₹20,000, starting this Friday"*.

## Test

```bash
.venv/Scripts/python.exe -m unittest -v     # offline: no network, no API key needed
.venv/Scripts/python.exe smoke.py           # live: one real call per outside service
```

## Editing the site

The journey pages are generated. Edit `super-travel/journeys.json` (or `build.py`), then run:

```bash
python super-travel/build.py
```

Do the same after editing `styles.css` or `main.js`. The tests fail if the generated pages are stale.

## Deploying

- **Site:** any static host. On Vercel, import the repo, set the Root Directory to `super-travel`, and leave the build command empty.
- **Concierge:** needs a server that stays running, so it cannot run on Vercel. Streamlit Community Cloud works: entry file `app.py`, and put your keys in its Secrets settings.
- **Link them:** the site's "Plan a trip" buttons use the `chat-url` meta tag in `super-travel/index.html`. Set it to the deployed concierge URL and run `build.py`.

The API has no login and spends your model quota, so do not expose it publicly without adding a guard.

## More documentation

- [backend-skeleton.md](backend-skeleton.md): concierge, tools, schemas, API, config, tests
- [frontend-skeleton.md](frontend-skeleton.md): site and Streamlit UI, design tokens, generated files
- `requirements.md`, `build-plan.md`, `roadmap.md`: the original plan, now outdated
