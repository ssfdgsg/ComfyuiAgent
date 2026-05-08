FROM yanwk/comfyui-boot:latest

LABEL maintainer="comfyui-agent"

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ────────────────────────────────────────────────────────────
COPY requirements.txt /opt/comfyui-agent/requirements.txt
RUN pip install --no-cache-dir -r /opt/comfyui-agent/requirements.txt

# ── Copy agent source ──────────────────────────────────────────────────────
COPY agent.py   /opt/comfyui-agent/agent.py
COPY config.py  /opt/comfyui-agent/config.py
COPY tools/     /opt/comfyui-agent/tools/
COPY llm/       /opt/comfyui-agent/llm/
COPY web/       /opt/comfyui-agent/web/
COPY workflows/ /opt/comfyui-agent/workflows/

RUN mkdir -p /opt/comfyui-agent/state

# ── Startup hook ──────────────────────────────────────────────────────────
# yanwk/comfyui-boot executes scripts in /docker/scripts/60_start_comfy/
COPY docker/agent-start.sh /docker/scripts/60_start_comfy/99_start_agent.sh
RUN chmod +x /docker/scripts/60_start_comfy/99_start_agent.sh

WORKDIR /opt/comfyui-agent

# Expose: 8188=ComfyUI, 8080=Agent Web UI
EXPOSE 8188 8080

# Default: start the web server (ComfyUI is started by the boot script)
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8080"]
