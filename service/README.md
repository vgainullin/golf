# research-runs

Backend and CLI for a donate-to-run research service. You pick a project and
donate. Every full run price in the project's pool (the price is $20 for golf)
pays for one research run. The project's research agent plans the run, the run
gets built and scored on a fixed benchmark, and the score lands on the project's
leaderboard with a report from the agent on what it found and what to try next.

golf is the first project. A golf run produces one candidate player, either
Python agent code or a DQN trained with `src.tournament`. The candidate plays as
`C` in the seat-cycled roster `C,L,I,R` (Bayes lookahead, improved heuristic,
random): all 24 seatings, 1000 games each, 9 holes. The score is its average
points per hole, and lower is better. The DQN champion's row in the top-level
README came from the same setup (it sat as `D` in `L,D,I,R`), so the board
starts with two baselines: the Bayes lookahead player (the champion) at 8.751
and the Exp14 DQN at 9.582.

Before each run, the research agent reads golf's experiment journal
(`docs/experiments.md`, `docs/beyond-heuristic-rl.md`, `data/llm_benchmarks.md`,
fetched from the repo at plan time) and the full report of every earlier run on
the board. From that it decides what to try: a new agent of its own design, a
change to the current leader's code, or a CPU training run. The leader's source
is included as the bar to beat. Its review is saved with the run and printed in
the report.

Payments are stubbed. A donation is recorded as captured and nothing is charged.

## Layout

```
service/
  src/            Cloudflare Worker: HTTP API, run lifecycle, research agent, dispatchers
  migrations/     D1 schema and the golf project seed
  cli/research.mjs  CLI over the API (no dependencies), includes a pull runner
  runner/execute.py Executes one run spec in a golf checkout and writes a result JSON
  agents/         The lookahead as agent code, plus the scripted one-change variants
  scripts/e2e.sh  Local end-to-end check
../.github/workflows/research_run.yml   GitHub Actions executor (CPU)
```

## Run lifecycle

```
donation ──► funded ──plan──► planned ──dispatch──► dispatched ──► running ──result──► evaluated ──analyze──► reported
                (agent writes the spec)                  (runner or workflow)                 (agent writes the report)
```

- A donation goes into the project pool. Each time the pool reaches the run price, a run is created and the price is taken out.
- A donor who pays for a full run can submit their own candidate (agent code or a checkpoint URL). That skips planning.
- Each step is a guarded status update (`UPDATE ... WHERE status = ?`), so the cron and request handlers can't take the same step twice. `planning` and `analyzing` act as locks and get released after 15 minutes if a worker dies. Failed steps retry up to three times.
- The planner and the reporter use Claude (`AGENT_MODEL`, default `claude-opus-5-5`) when `ANTHROPIC_API_KEY` is set. The journal is the first, cached system block, so planner and reporter calls share it. Without a key nothing reads the journal: the fallback cycles through two fixed lookahead variants in `agents/`, and the report is a template.

## Dispatchers

The Worker never trains or evaluates anything itself. `DISPATCHER` picks who does:

| value | what happens | use |
|---|---|---|
| `mock` | Fake numbers come back immediately. The rows are marked simulated. | Demoing the loop, UI work |
| `queue` | The run waits until a runner claims it with `POST /v1/runner/claim`. | Local machine or a GPU box you control (`research runner`) |
| `github` | Dispatches `research_run.yml` on `GITHUB_REPO`. The workflow fetches the spec and posts the result. | Hosted CPU runs |

GPU runs are refused unless `ALLOW_GPU=true`, and the workflow refuses them as
well because they aren't wired to the Lambda Labs path yet. Nothing in this
service launches paid instances.

## Local quickstart

```bash
cd service
npm install
cp .dev.vars.example .dev.vars
npm run db:migrate:local
npm run dev                         # http://localhost:8787, DISPATCHER=mock

# in another shell
export RESEARCH_API=http://localhost:8787 ADMIN_TOKEN=dev-admin-token RUNNER_TOKEN=dev-runner-token
node cli/research.mjs projects
node cli/research.mjs donate golf --amount 20 --donor me
node cli/research.mjs tick          # or wait for the cron
node cli/research.mjs leaderboard golf
node cli/research.mjs runs golf
node cli/research.mjs report RUN_ID
```

To run real evaluations locally, start the Worker with `--var DISPATCHER:queue`
and start a runner from the repo root's venv (`uv sync` first):

```bash
node cli/research.mjs runner                       # full protocol (slow on CPU)
node cli/research.mjs runner --games-per-perm 20   # quick check; lands on board seatcycle-v1@20g
```

When the runner overrides the game count, the score goes on its own board
(`seatcycle-v1@20g`) so it can't be compared with full-protocol rows.

To submit your own candidate:

```bash
node cli/research.mjs donate golf --amount 20 --code my_agent.py --title "My agent"
```

`my_agent.py` defines `stage0(state, seat)` and `stage1(state, seat)`. Each one
returns an `(N,)` long tensor of actions for N parallel games. It can also define
`reset(state, seat)` (called at the start of each hole) and `observe(state, seat)`
(called after every step) to track beliefs. See `agents/` for examples.

`npm run e2e` runs the whole loop against a fresh local database with the mock
dispatcher. `DISPATCHER=queue GAMES=20 BOARD=seatcycle-v1@20g npm run e2e` runs it
with a real evaluation.

## API

| method | path | auth | notes |
|---|---|---|---|
| GET | `/v1/projects` | | |
| GET | `/v1/projects/:slug` | | best entry, run counts |
| POST | `/v1/projects/:slug/donations` | | `{donor, amount_usd, note?, submission?: {title?, hypothesis?, spec: {kind, agent_code \| checkpoint}}}` |
| GET | `/v1/projects/:slug/leaderboard` | | `?protocol=` for other boards |
| GET | `/v1/projects/:slug/runs` | | |
| GET | `/v1/runs/:id` | | spec, result, report, events |
| GET | `/v1/runs/:id/report` | | markdown |
| POST | `/v1/runner/claim` | RUNNER_TOKEN | next queued run, or 204 |
| GET | `/v1/runner/runs/:id/spec` | RUNNER_TOKEN | |
| POST | `/v1/runner/runs/:id/started` | RUNNER_TOKEN | `{external_ref?}` |
| POST | `/v1/runner/runs/:id/result` | RUNNER_TOKEN | the JSON `execute.py` writes |
| POST | `/v1/admin/tick` | ADMIN_TOKEN | advance every runnable run now |
| POST | `/v1/admin/projects/:slug/baselines` | ADMIN_TOKEN | `{label, name}` scores an existing player (e.g. `L`) in the candidate seat as a baseline row |

## Deploying to Cloudflare

```bash
npx wrangler d1 create research-runs      # put the id in wrangler.toml
npm run db:migrate:remote
npx wrangler secret put RUNNER_TOKEN
npx wrangler secret put ADMIN_TOKEN
npx wrangler secret put ANTHROPIC_API_KEY
npx wrangler secret put GITHUB_TOKEN      # actions:write on vgainullin/golf, for DISPATCHER=github
npm run deploy
```

For the `github` dispatcher, set `DISPATCHER = "github"` and `PUBLIC_URL` in
`wrangler.toml`, and add the same runner token to the golf repo as the
`RESEARCH_RUNNER_TOKEN` Actions secret.

## Known gaps

- Payments are stubbed. Stripe Checkout plus a webhook that calls `donate()` would replace the stub.
- Planning and reporting run from the cron or `tick`. A planner call can outlast `waitUntil` after a donation request, so a run can sit in `funded` for up to a minute. Cloudflare Queues or Workflows would fit better.
- Submitted agent code runs unsandboxed inside the executor. The GitHub workflow keeps secrets out of that step and doesn't persist the checkout token, but a real deployment needs a proper sandbox.
- There's no GPU path. It would reuse `deploy/` and the Lambda Labs workflow, gated on `ALLOW_GPU`.
- Leaderboard ranks use raw means with no confidence intervals, so small gaps between rows may be noise.
- The Claude planner and reporter haven't been exercised against the real API yet because there was no API key in the dev environment. Their request and response handling was checked against a local mock of the Messages API; the e2e test covers the scripted path.
