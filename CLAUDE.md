# GlobeTrot

A travel concierge: a static marketing site (`super-travel/`, port 5173) plus a Streamlit chatbot (repo root, port 8501) that plans weather-aware, budget-checked itineraries using LLM tool calls.

## Read these first

Before planning or changing anything, read both reference files in full:

- [backend-skeleton.md](backend-skeleton.md): concierge, tools, schemas, config, tests, content store, planned HTTP API.
- [frontend-skeleton.md](frontend-skeleton.md): site and Streamlit UI, design tokens, components, generated-vs-hand-written files.

Read the one that matches the task, plus the other whenever the change crosses the site/backend boundary (a new site feature that needs data, a schema change that affects rendering, a new tool that shows in the UI). When unsure, read both.

## Keep them current

After any change to the site or backend, update the affected skeleton file in the same change set. Each file's §0 has a table of "if you changed X, update section Y". In short:

- Update contracts, layout and status markers (✅ built, 🟡 partial, ⬜ planned) to match what you actually did.
- Only mark ✅ after it has been run or checked. Say what was and was not verified.
- Add one line to the change log (§12) with the date, what changed and why.
- If a skeleton disagrees with the code, the code wins: fix the skeleton and log it.
- Do not delete decisions in §11; mark them superseded.

## Commands

```bash
.venv/Scripts/python.exe -m streamlit run app.py --server.port 8501   # concierge
python super-travel/serve.py 5173                                     # site
python super-travel/build.py                                          # regenerate site pages
.venv/Scripts/python.exe -m unittest -v                               # all offline tests
```

## Rules that are easy to break

- Never hand-edit generated files: `super-travel/journeys/**/index.html` and the cards between the `journeys:start/end` markers in `index.html`. Edit `journeys.json` or `build.py`, then run `build.py`. This also applies after editing `styles.css` or `main.js`.
- Tools in `tools.py` never raise; they return `{"error": "..."}`.
- Budget totals are recomputed in Python (`_reconcile`), never trusted from the model.
- Escape all user and model text with `render.esc` before putting it in HTML.
- Design tokens live in three places (`super-travel/styles.css`, `theme.css`, `.streamlit/config.toml`); change them together.
- Secrets stay in `.env`; never print or commit them.
- Add or update a test with every behaviour change.
- `requirements.md`, `build-plan.md` and `roadmap.md` are outdated; the skeleton files and the code are current.
