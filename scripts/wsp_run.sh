#!/usr/bin/env bash
# Weight-space programming run: headless Claude Code (Opus) edits the DQN champion's
# weights with a fixed forward-pass budget, then final.pt gets the held-out benchmark.
#   scripts/wsp_run.sh <champion.pt> [budget=25] [run_name] [effort=high]
# Continue from an earlier run: pass its final.pt as <champion.pt> and set PRIOR_RUN=<that run dir>
# (its notes and scripts are copied in as reference). PRACTICE_SEED (default 1) picks the practice decks.
# shortcut: the pass budget and the no-training rule are enforced by prompt and audited in
#   transcript.jsonl, not sandboxed; move to a container if a run ever cheats.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${1:?champion checkpoint}; BUDGET=${2:-25}; NAME=${3:-run_$(date +%Y%m%d_%H%M%S)}; EFFORT=${4:-high}
RUN=data/wsp/$NAME
mkdir -p "$RUN"
# The champion was saved on MPS; re-save on CPU so a plain torch.load works.
.venv/bin/python -c "import sys,torch; torch.save(torch.load(sys.argv[1],map_location='cpu',weights_only=False),sys.argv[2])" "$CKPT" "$RUN/base.pt"
echo "{\"max_passes\": $BUDGET, \"practice_seed\": ${PRACTICE_SEED:-1}}" > "$RUN/budget.json"
PRIOR_TEXT=""
if [ -n "${PRIOR_RUN:-}" ]; then
  mkdir -p "$RUN/prior"
  cp "$PRIOR_RUN"/NOTES.md "$PRIOR_RUN"/*.py "$RUN/prior/"
  cp "$PRIOR_RUN"/circuit_meta*.json "$RUN/prior/" 2>/dev/null || true
  PRIOR_TEXT=$(sed "s#{RUN_DIR}#$RUN#g" scripts/wsp_prompt_prior.md)
fi
echo "{\"effort\": \"$EFFORT\", \"commit\": \"$(git rev-parse --short HEAD)\"}" > "$RUN/run_meta.json"
PRIOR_TEXT="$PRIOR_TEXT" .venv/bin/python -c "
import os, sys
t = open('scripts/wsp_prompt.md').read()
p = os.environ['PRIOR_TEXT']
t = t.replace('{PRIOR}', p + '\n\n' if p else '').replace('{RUN_DIR}', sys.argv[1]).replace('{BUDGET}', sys.argv[2])
open(sys.argv[1] + '/prompt.md', 'w').write(t)" "$RUN" "$BUDGET"

taskset -c 0-3 env -u CLAUDE_EFFORT claude -p "$(cat "$RUN/prompt.md")" --model opus --effort "$EFFORT" \
  --output-format stream-json --verbose \
  --allowedTools "Bash Read Write Edit Glob Grep" --disallowedTools "WebFetch WebSearch Agent" \
  > "$RUN/transcript.jsonl" 2> "$RUN/claude.err" || echo "claude exited $?" >> "$RUN/claude.err"

[ -f "$RUN/final.pt" ] || { echo "no final.pt in $RUN"; exit 1; }
# Held-out benchmark (seed 0, never seen by the programmer) for both, so the comparison is paired.
for m in base final; do
  echo "{\"kind\":\"checkpoint\",\"checkpoint\":{\"url\":\"$RUN/$m.pt\"},\"protocol\":\"seatcycle-v1\"}" > "$RUN/heldout_$m.spec.json"
  taskset -c 0-3 env OMP_NUM_THREADS=4 .venv/bin/python -m service.runner.execute \
    --spec "$RUN/heldout_$m.spec.json" --out "$RUN/heldout_$m.json"
done
