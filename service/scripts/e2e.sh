#!/usr/bin/env bash
# End-to-end check against a local Worker with a fresh local D1.
#   DISPATCHER=mock  (default) donate -> plan -> simulated result -> leaderboard -> report
#   DISPATCHER=queue           same, but a real runner executes the run (needs the golf venv)
set -euo pipefail
cd "$(dirname "$0")/.."
DISPATCHER="${DISPATCHER:-mock}"
PORT="${PORT:-8799}"
export RESEARCH_API="http://127.0.0.1:$PORT" RUNNER_TOKEN=dev-runner-token ADMIN_TOKEN=dev-admin-token
if curl -sf "$RESEARCH_API/v1/projects" >/dev/null 2>&1; then echo "port $PORT is already serving; stop it first" >&2; exit 1; fi
STATE="$(mktemp -d)"
# wrangler ignores SIGTERM from a non-interactive shell and keeps workerd alive, so take down the whole tree.
kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -9 "$1" 2>/dev/null || true; }
cleanup() { [ -n "$WPID" ] && kill_tree "$WPID"; if [ -n "${KEEP:-}" ]; then echo "state kept in $STATE"; else rm -rf "$STATE"; fi; }
WPID=""
trap cleanup EXIT

npx wrangler d1 migrations apply research-runs --local --persist-to "$STATE" >/dev/null
npx wrangler dev --port "$PORT" --persist-to "$STATE" --var "DISPATCHER:$DISPATCHER" \
  --var RUNNER_TOKEN:dev-runner-token --var ADMIN_TOKEN:dev-admin-token --var STRIPE_WEBHOOK_SECRET:whsec_dev >"$STATE/wrangler.log" 2>&1 &
WPID=$!
for _ in $(seq 60); do curl -sf "$RESEARCH_API/v1/projects" >/dev/null && break; sleep 1; done

R() { node cli/research.mjs "$@"; }
R projects
R donate golf --amount 12 --donor alice --note "half a run"
R donate golf --amount 28 --donor bob --note "try something with the lookahead player"
if [ "$DISPATCHER" = mock ]; then
  printf 'from src.vectorized_golf import heuristic_stage0, heuristic_stage1\ndef stage0(state, seat):\n    return heuristic_stage0(state, seat)\ndef stage1(state, seat):\n    return heuristic_stage1(state, seat)\n' >"$STATE/agent.py"
  R donate golf --amount 20 --donor carol --code "$STATE/agent.py" --title "Carol's base heuristic"
  if R donate golf --amount 5 --code "$STATE/agent.py" 2>/dev/null; then echo "expected a partial-run submission to be refused" >&2; exit 1; fi
  if RUNNER_TOKEN=wrong R runner --once 2>/dev/null; then echo "expected a bad runner token to be refused" >&2; exit 1; fi
  if ADMIN_TOKEN= R donate golf --amount 20 2>/dev/null; then echo "expected a donation without ADMIN_TOKEN to be refused" >&2; exit 1; fi
  # A paid Stripe Checkout session funds one run; Stripe redelivering it funds nothing.
  stripe() { node scripts/stripe-event.mjs "$1" "$2" | sh; }
  stripe cs_test_e2e 2000 | grep -q '"credited golf"' || { echo "expected the Stripe payment to be credited" >&2; exit 1; }
  stripe cs_test_e2e 2000 | grep -q '"duplicate"' || { echo "expected a redelivered Stripe event to be ignored" >&2; exit 1; }
  if curl -sf -X POST "$RESEARCH_API/v1/webhooks/stripe" -H 'stripe-signature: t=1,v1=00' -d '{}' >/dev/null; then echo "expected a bad Stripe signature to be refused" >&2; exit 1; fi
fi
R tick
if [ "$DISPATCHER" = queue ]; then
  R runner --once ${GAMES:+--games-per-perm $GAMES}
fi
R tick
R runs golf
R leaderboard golf --protocol "${BOARD:-seatcycle-v1}"
RUN=$(R runs golf --json | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const r=JSON.parse(s).filter(r=>r.status==="reported");if(!r.length){console.error("no reported run");process.exit(1)}console.log(r[0].id)})')
R report "$RUN"
echo "e2e ok ($DISPATCHER)"
