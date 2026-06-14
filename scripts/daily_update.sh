#!/usr/bin/env bash
# Daily refresh: pull latest international results, re-score predictions, settle paper + Kelly
# portfolio against real Kalshi results, regenerate STANDINGS.md, and push to GitHub.
# Scheduled by launchd (see scripts/com.wc2026.dailyupdate.plist). Logs to ~/Library/Logs.
set -uo pipefail

REPO="/Users/sansarkarki/Documents/World Cup"
LOG="$HOME/Library/Logs/wc2026_daily.log"        # outside the repo so it isn't committed
PY="$REPO/.venv/bin/python"

cd "$REPO" || { echo "repo not found" >> "$LOG"; exit 1; }

{
  echo "================ daily update $(date) ================"
  # load API keys (api-football / SharpAPI) if present — for live settlement & anchors
  if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi

  "$PY" scripts/fetch_data.py        || echo "WARN: fetch_data failed (continuing)"
  "$PY" src/track.py --days 4        || echo "WARN: track failed"

  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "daily standings update $(date +%F)" >/dev/null 2>&1
    if git push origin main >/dev/null 2>&1; then
      echo "OK: pushed standings"
    else
      echo "ERROR: git push failed (check 'gh auth status' / network)"
    fi
  else
    echo "no changes to commit"
  fi
  echo "================ done $(date) ================"
} >> "$LOG" 2>&1
