# yanwk/comfyui-boot ships a startup-script system under /docker/scripts/.
# We layer the agent on top without touching ComfyUI itself.

FROM yanwk/comfyui-boot:latest

LABEL maintainer="comfyui-agent"

# ── system deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps for the agent ─────────────────────────────────────────────
COPY requirements.txt /opt/comfyui-agent/requirements.txt
RUN pip install --no-cache-dir -r /opt/comfyui-agent/requirements.txt

# ── Copy agent source ─────────────────────────────────────────────────────
COPY agent.py       /opt/comfyui-agent/agent.py
COPY config.py      /opt/comfyui-agent/config.py
COPY tools/         /opt/comfyui-agent/tools/
COPY workflows/     /opt/comfyui-agent/workflows/

# Create the state directory that the agent writes to
RUN mkdir -p /opt/comfyui-agent/state

# ── Startup hook ─────────────────────────────────────────────────────────
# yanwk/comfyui-boot runs all *.sh scripts in /docker/scripts/60_start_comfy/
# We add our own startup script that launches the agent after ComfyUI is up.
COPY docker/agent-start.sh /docker/scripts/60_start_comfy/99_start_agent.sh
RUN chmod +x /docker/scripts/60_start_comfy/99_start_agent.sh

WORKDIR /opt/comfyui-agent

# Default: run the REPL (override CMD to pass a one-shot task)
CMD ["python", "agent.py"]
