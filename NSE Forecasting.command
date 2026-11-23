#!/bin/bash
# Double-click this file in Finder to open the NSE Forecasting app in your browser.
# A Terminal window opens and runs the app; close that window to stop it.
cd "$(dirname "$0")" || exit 1
URL="http://127.0.0.1:8766/"

if curl -s -o /dev/null -m 2 "${URL}api/status"; then
  echo "The app is already running; opening it."
  open "$URL"
  exit 0
fi

UV=$(command -v uv)
for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
  [ -n "$UV" ] && break
  [ -x "$candidate" ] && UV=$candidate
done
if [ -z "$UV" ]; then
  echo "Could not find uv (https://docs.astral.sh/uv/). Press Return to close."
  read -r
  exit 1
fi

exec "$UV" run --extra kite --extra slm forecasting ui
