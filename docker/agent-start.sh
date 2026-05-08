#!/usr/bin/env bash
# Startup hook for yanwk/comfyui-boot.
# Waits for ComfyUI to become ready, then launches the agent if
# AGENT_TASK is set (one-shot) or AGENT_REPL=1 (interactive REPL via stdin).

set -e

AGENT_DIR="/opt/comfyui-agent"
COMFYUI_URL="${COMFYUI_URL:-http://localhost:8188}"
MAX_WAIT=120   # seconds

echo "[agent-start] Waiting for ComfyUI at ${COMFYUI_URL} ..."
waited=0
until curl -sf "${COMFYUI_URL}/system_stats" > /dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -ge $MAX_WAIT ]; then
        echo "[agent-start] ComfyUI did not start within ${MAX_WAIT}s — skipping agent launch."
        exit 0
    fi
done
echo "[agent-start] ComfyUI is ready (waited ${waited}s)."

# Run an initial resource scan so state/resources.md exists from the start
cd "$AGENT_DIR"
python -c "from tools.state import refresh_resources_doc; refresh_resources_doc()" || true

if [ -n "${AGENT_TASK}" ]; then
    echo "[agent-start] Running one-shot task: ${AGENT_TASK}"
    python agent.py "${AGENT_TASK}"
elif [ "${AGENT_REPL:-0}" = "1" ]; then
    echo "[agent-start] Starting agent REPL ..."
    python agent.py
else
    echo "[agent-start] No AGENT_TASK or AGENT_REPL set — agent is available but not auto-started."
    echo "              Run: docker exec -it <container> python /opt/comfyui-agent/agent.py"
fi
