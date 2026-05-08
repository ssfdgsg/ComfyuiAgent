"""
FastAPI web server for ComfyUI Agent.

Endpoints:
  GET  /                    → index.html
  GET  /api/status          → ComfyUI + agent status
  GET  /api/workflow        → current workflow (nodes + summary)
  PATCH /api/workflow/node  → inline-edit a node's parameters
  DELETE /api/workflow/node/{id} → remove a node
  POST /api/workflow/execute → queue to ComfyUI
  GET  /api/resources       → installed models + nodes
  POST /api/settings        → hot-update API key / provider
  WS   /ws                  → streaming agent chat

Start: python web/app.py  (or uvicorn web.app:app --port 8080)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import WEB_HOST, WEB_PORT, COMFYUI_URL, DEFAULT_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY
from agent import Agent
from llm.base import MediaFile
from tools.comfyui_api import get_client
from tools.state import read_resources_doc, refresh_resources_doc

app = FastAPI(title="ComfyUI Agent", version="2.0")

STATIC_DIR = Path(__file__).parent / "static"

# One shared agent instance (single-user model for simplicity)
_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


# ─────────────────────────────────────────────── static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─────────────────────────────────────────────── status
@app.get("/api/status")
async def get_status():
    c = get_client()
    agent = get_agent()
    comfyui_ok = await asyncio.to_thread(c.is_alive)
    return {
        "comfyui": "online" if comfyui_ok else "offline",
        "comfyui_url": COMFYUI_URL,
        "provider": agent._provider_name,
        "model": agent._model or "(default)",
        "workflow_nodes": agent.core.wf.node_count(),
    }


# ─────────────────────────────────────────────── workflow REST
@app.get("/api/workflow")
async def get_workflow():
    agent = get_agent()
    return {
        "nodes": agent.core.workflow_nodes_json(),
        "summary": agent.core.wf.get_summary(),
    }


@app.patch("/api/workflow/node")
async def patch_node(payload: dict):
    """
    Inline-edit a node's parameters from the web UI.
    Body: {"node_id": "5", "inputs": {"steps": 30, "cfg": 7.5}}
    """
    agent = get_agent()
    node_id = str(payload.get("node_id", ""))
    inputs = payload.get("inputs", {})
    if not node_id:
        return JSONResponse({"error": "node_id required"}, status_code=400)
    ok = agent.core.wf.update_node_inputs(node_id, inputs)
    if ok:
        agent.core._save_state("web_patch_node", {"node_id": node_id, "fields": list(inputs.keys())})
    return {"success": ok, "nodes": agent.core.workflow_nodes_json()}


@app.delete("/api/workflow/node/{node_id}")
async def delete_node(node_id: str):
    agent = get_agent()
    ok = agent.core.wf.remove_node(node_id)
    if ok:
        agent.core._save_state("web_remove_node", {"node_id": node_id})
    return {"success": ok, "nodes": agent.core.workflow_nodes_json()}


@app.post("/api/workflow/execute")
async def execute_workflow():
    agent = get_agent()
    c = get_client()
    if not await asyncio.to_thread(c.is_alive):
        return JSONResponse({"error": "ComfyUI not reachable"}, status_code=503)
    result = await asyncio.to_thread(c.queue_prompt, agent.core.wf.to_api_format())
    return result


@app.post("/api/workflow/save")
async def save_workflow(payload: dict = None):
    agent = get_agent()
    name = (payload or {}).get("name", "")
    saved = agent.core.wf.save(name=name or None)
    return {"saved": saved}


# ─────────────────────────────────────────────── resources
@app.get("/api/resources")
async def get_resources():
    content = await asyncio.to_thread(read_resources_doc)
    return {"markdown": content}


@app.post("/api/resources/refresh")
async def refresh_resources():
    content = await asyncio.to_thread(refresh_resources_doc)
    return {"markdown": content}


# ─────────────────────────────────────────────── settings
@app.get("/api/settings")
async def get_settings():
    agent = get_agent()
    return {
        "provider": agent._provider_name,
        "model": agent._model,
        "base_url": agent._base_url,
        "has_gemini_key": bool(agent._api_key or GEMINI_API_KEY),
        "has_openai_key": bool(OPENAI_API_KEY),
    }


@app.post("/api/settings")
async def update_settings(payload: dict):
    """
    Hot-update provider settings.
    Body keys: provider, api_key, model, base_url
    """
    agent = get_agent()
    agent.update_settings(
        provider=payload.get("provider", ""),
        api_key=payload.get("api_key", ""),
        model=payload.get("model", ""),
        base_url=payload.get("base_url", ""),
    )
    return {"ok": True, "provider": agent._provider_name, "model": agent._model}


# ─────────────────────────────────────────────── WebSocket chat
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    agent = get_agent()
    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            msg_type = payload.get("type", "message")

            if msg_type != "message":
                continue

            content = payload.get("content", "")
            files_raw = payload.get("files", [])

            # Decode any attached media files
            media_files: list[MediaFile] = []
            for f in files_raw:
                try:
                    data = base64.b64decode(f["b64"])
                    media_files.append(MediaFile(
                        name=f.get("name", "file"),
                        mime_type=f.get("mime", "application/octet-stream"),
                        data=data,
                    ))
                except Exception:
                    pass

            # Stream agent response
            async for event in agent.run_stream(content, media_files=media_files or None):
                await ws.send_text(json.dumps(event, ensure_ascii=False))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "content": str(e)}))
        except Exception:
            pass


# ─────────────────────────────────────────────── main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host=WEB_HOST, port=WEB_PORT, reload=False)
