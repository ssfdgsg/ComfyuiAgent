#!/usr/bin/env bash
# Startup hook for yanwk/comfyui-boot.
# Waits for ComfyUI, then starts the Agent Web UI.

set -e

AGENT_DIR="/opt/comfyui-agent"
COMFYUI_URL="${COMFYUI_URL:-http://localhost:8188}"
MAX_WAIT=120

echo "[agent] Waiting for ComfyUI at ${COMFYUI_URL} ..."
waited=0
until curl -sf "${COMFYUI_URL}/system_stats" > /dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -ge $MAX_WAIT ]; then
        echo "[agent] ComfyUI not ready after ${MAX_WAIT}s — starting agent anyway."
        break
    fi
done
echo "[agent] ComfyUI ready (waited ${waited}s). Starting Agent Web UI on :${WEB_PORT:-8080}..."

cd "$AGENT_DIR"
# Initial resource scan
python -c "from tools.state import refresh_resources_doc; refresh_resources_doc()" || true

# Start the web server (keep running in foreground so the container stays alive)
exec python -m uvicorn web.app:app \
    --host "${WEB_HOST:-0.0.0.0}" \
    --port "${WEB_PORT:-8080}" \
    --log-level info
