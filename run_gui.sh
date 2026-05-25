#!/usr/bin/env bash
# ===========================================================================
# PEMFC Simulator — one-click launcher for the Streamlit GUI (macOS / Linux)
# ===========================================================================
# Run with:   bash run_gui.sh
# A browser tab opens at http://localhost:8501. Press Ctrl+C to stop.
# ===========================================================================

set -e
cd "$(dirname "$0")"

if command -v streamlit >/dev/null 2>&1; then
    ST="streamlit"
elif [ -x ".venv/bin/streamlit" ]; then
    ST=".venv/bin/streamlit"
else
    echo "Could not locate streamlit. Install it once with:"
    echo "    pip install streamlit"
    exit 1
fi

echo "Starting PEMFC Simulator GUI ..."
echo "  Streamlit : $ST"
echo "  App       : $(pwd)/gui/app.py"
echo
echo "(Your browser will open automatically. Press Ctrl+C to stop.)"
echo

exec "$ST" run gui/app.py
