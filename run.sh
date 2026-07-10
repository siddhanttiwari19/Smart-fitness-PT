#!/bin/bash
# AI Squat Coach launcher
# Usage: bash run.sh    (or chmod +x run.sh && ./run.sh)

set -e
cd "$(dirname "$0")"

# Prefer explicit python3; fall back to PATH
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# Quick dependency check — fail early with a clear message
$PYTHON - <<'PY' 2>/dev/null || {
    echo "Installing missing dependencies…"
    $PYTHON -m pip install -q -r requirements.txt
}
import cv2, mediapipe, numpy  # noqa
PY

exec $PYTHON main.py "$@"
