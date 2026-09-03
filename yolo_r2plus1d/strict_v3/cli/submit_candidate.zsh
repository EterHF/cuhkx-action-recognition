#!/bin/zsh
# Submit exactly one previously validated candidate.  The quota guard avoids
# accidentally spending a slot when Kaggle reports no remaining submissions.
set -euo pipefail

if (( $# != 1 )); then
  print "usage: $0 path/to/submission.csv" >&2
  exit 2
fi
candidate="$1"
if [[ ! -f "$candidate" ]]; then
  print "candidate not found: $candidate" >&2
  exit 2
fi

env_python=".conda/envs/cuhkx/bin/python3.11"
kaggle_bin=".conda/envs/cuhkx/bin/kaggle"
# Generate a fresh OAuth access token from the locally authorized refresh token.
# The Kaggle CLI's cached expiration grace period can otherwise leave a stale
# token in the environment after a long quota wait.
token="$("$kaggle_bin" auth print-access-token 2>/dev/null || true)"
"$env_python" - "$candidate" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 405 or set(rows[0]) != {"path", "prediction"}:
    raise SystemExit(f"invalid submission: rows={len(rows)} fields={set(rows[0]) if rows else None}")
if any(not row["path"].startswith("small_model_track_test/") for row in rows):
    raise SystemExit("invalid path prefix")
if len({row["path"] for row in rows}) != 405:
    raise SystemExit("duplicate submission paths")
print(f"validated={path} rows={len(rows)}")
PY

if [[ -n "$token" ]]; then
  limits="$(KAGGLE_API_TOKEN="$token" "$kaggle_bin" competitions submission-limits -c cuhk-x-competition-small-model-track)"
else
  # Fall back to the CLI's OS-backed OAuth credential when token refresh is
  # temporarily rate-limited.
  limits="$($kaggle_bin competitions submission-limits -c cuhk-x-competition-small-model-track)"
fi
remaining="$(print -r -- "$limits" | sed -n 's/^Remaining today: //p')"
if [[ -z "$remaining" || "$remaining" == 0 ]]; then
  print -r -- "$limits" >&2
  print "submission blocked: no remaining daily quota" >&2
  exit 3
fi

if [[ -n "$token" ]]; then
  KAGGLE_API_TOKEN="$token" "$kaggle_bin" competitions submit \
    -c cuhk-x-competition-small-model-track \
    -f "$candidate" \
    -m "validated candidate: ${candidate:t}"
else
  "$kaggle_bin" competitions submit \
    -c cuhk-x-competition-small-model-track \
    -f "$candidate" \
    -m "validated candidate: ${candidate:t}"
fi
