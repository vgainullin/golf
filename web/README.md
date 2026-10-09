# Golf Agents Lab (web UI)

A static site for playing Golf against the repo's agents, running simulation competitions, reading the research notes, and crowdfunding new experiments. No build step and no server code: plain HTML, CSS and ES modules.

## Run it locally

Serve the **repository root** (the research view reads Markdown from `docs/` and `data/`):

```bash
python -m http.server 8000
# open http://localhost:8000/web/
```

## Pages

| Route | What it does |
|---|---|
| `#/` | Leaderboard from `data/leaderboard.json`, plus how to benchmark your own agent and the method. |
| `#/play` | Play a match against three AI seats, with an optional Bayes "coach" hint. |
| `#/arena` | Pick four agents and run seat-cycled matches in a Web Worker. Results include 95% CIs and copy out as Markdown. |
| `#/research` | Research agenda and notes. Notes listed in `data/notes.json` are rendered from the repo's Markdown. |
| `#/fund` | Donation tiers and per-experiment funding goals from `data/funding.json`. |

## Agents in the browser

`js/engine.js` is a port of the rules in `src/vectorized_golf.py` and the turn loop in `src/bayes_optimal.py`; `js/agents.js` ports the random, greedy, base heuristic, improved heuristic and Bayes 1-step lookahead players. With the same opponents, the browser lookahead scores match the Python player:

| Config | Python (`src.bayes_optimal --player lookahead`, 3000 games) | Browser (3000 games) |
|---|---|---|
| L,R,I,R | 6.869 | 6.866 |
| L,I,R,I | 7.844 | 7.835 |

The DQN champion and LLM players still run only in Python.

Tests: `node --test web/tests/*.test.mjs`

## Donations

Donations are **not live**. To enable them, put a hosted checkout link (a Stripe Payment Link, Open Collective, GitHub Sponsors, ...) in `data/funding.json` under `provider.donate_url`. If the provider accepts a preset amount as a query parameter, set `provider.amount_param`. The "Back this" buttons append `client_reference_id=<project id>` so donations can be attributed to a project. Raised totals are edited by hand; the site never handles payment details.

## Adding content

- **Leaderboard row:** append to `data/leaderboard.json` with a `source` pointing at the result file in the repo.
- **Research note:** add the Markdown file to the repo and list it in `data/notes.json`.
- **Research project:** add it to `projects` in `data/funding.json`; set `status: "done"` and `note: <note id>` once it's written up.

Third-party code: `vendor/marked.min.js` (marked 12.0.2, MIT, see `vendor/marked.LICENSE.md`).
