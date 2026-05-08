# ComfyUI Agent

A Python AI agent that runs **inside** the `yanwk/comfyui-boot` container and operates ComfyUI programmatically via natural language.

## What it can do

| Capability | Details |
|---|---|
| Workflow editing | Add / remove nodes, create / delete links, update parameters |
| One-click workflow creation | Describe a pipeline in plain language → agent builds the full workflow |
| Smart long-workflow handling | Sends only a compact summary to the LLM; fetches node details on demand |
| Model search & install | HuggingFace, CivitAI, ComfyUI Manager model registry |
| Custom node search & install | ComfyUI Manager custom node registry (git clone + pip install) |
| State documentation | `state/resources.md` and `state/workflow_state.md` updated after every operation |
| Execution | Queue workflows, poll for completion |

## Project layout

```
comfyui-agent/
├── agent.py              ← Main Claude-based agent loop
├── config.py             ← All paths and env-var config
├── tools/
│   ├── comfyui_api.py    ← HTTP client for ComfyUI REST API
│   ├── workflow.py       ← In-memory workflow (add/remove/link/patch)
│   ├── models.py         ← Model scanning, HuggingFace, CivitAI, downloading
│   ├── nodes.py          ← Custom node registry, install, update
│   └── state.py          ← state/resources.md + workflow_state.md + history.jsonl
├── state/                ← Auto-generated state documents (git-ignored)
├── workflows/            ← Saved workflow JSON files
├── docker/
│   └── agent-start.sh   ← Startup hook for yanwk/comfyui-boot
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick start

### 1. Environment variables

```bash
cp .env.example .env
# Edit .env — set at minimum ANTHROPIC_API_KEY
```

`.env.example`:
```
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_TOKEN=hf_...     # optional but needed for gated models
CIVITAI_API_KEY=...           # optional
AGENT_TASK=                   # optional: auto-run a task on container start
AGENT_REPL=0
```

### 2. Build and run

```bash
docker compose up --build
```

ComfyUI will be available at http://localhost:8188.

The agent starts after ComfyUI is ready. Default: no auto-task, agent waits.

### 3. Interact with the agent

```bash
# Interactive REPL
docker exec -it comfyui-agent python /opt/comfyui-agent/agent.py

# One-shot task
docker exec comfyui-agent python /opt/comfyui-agent/agent.py \
  "Create an SDXL text-to-image workflow with a ControlNet depth pass"

# Or set AGENT_TASK= in docker-compose.yml / .env to auto-run on start
```

### 4. Local dev (without Docker)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export COMFYUI_URL=http://localhost:8188
python agent.py
```

---

## Architecture: handling long workflows

ComfyUI workflows can contain 50–200+ nodes. Sending the full JSON to the LLM on every turn would waste tokens and slow things down.

**WorkflowManager** solves this with a two-level access pattern:

```
Claude sees:                            Claude can request:
┌─────────────────────────┐             ┌─────────────────────────┐
│ get_workflow_summary()  │  (always)   │ get_node_detail("42")   │
│                         │             │ get_nodes_by_type("K…") │
│  [1] CheckpointLoader   │             │                         │
│  [2] CLIPTextEncode     │  ──────────►│  full JSON for that     │
│  [3] CLIPTextEncode     │             │  subset only            │
│  ...  15 more nodes ... │             └─────────────────────────┘
└─────────────────────────┘
```

Mutations are applied **incrementally** via `apply_workflow_patch` — Claude sends a diff, not the whole workflow.

---

## State documents

After every mutating operation the agent writes:

| File | Contents |
|---|---|
| `state/resources.md` | All installed models (by category) and custom node packages |
| `state/workflow_state.md` | Current workflow summary + file path |
| `state/history.jsonl` | Append-only operation log (JSON lines) |

These files are included in the system prompt so the agent always knows what resources are available without a full rescan.

---

## Available tools (Claude sees these)

**Workflow**
- `get_workflow_summary` / `get_node_detail` / `get_nodes_by_type`
- `add_node` / `remove_node` / `update_node`
- `create_link` / `remove_link`
- `apply_workflow_patch` ← batch changes in one call
- `load_workflow` / `load_workflow_preset` / `save_workflow`
- `queue_workflow` / `get_execution_status`
- `list_available_node_types` / `get_node_type_info`

**Models**
- `list_local_models`
- `search_models_huggingface` / `search_models_civitai` / `search_comfyui_model_list`
- `get_huggingface_model_files`
- `download_model` (streaming, progress bar) / `download_model_background`
- `get_download_status`

**Custom Nodes**
- `list_installed_nodes`
- `search_custom_nodes`
- `install_custom_node` / `install_node_by_title` / `update_custom_node`

**State**
- `refresh_resources` / `read_resources`
- `read_workflow_state` / `read_operation_history`
- `list_saved_workflows`

---

## Example sessions

```
[You] Create a basic txt2img workflow for SDXL

→ tool: load_workflow_preset {"preset": "txt2img"}
← loaded 7-node workflow
→ tool: update_node {"node_id": "1", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}}
→ tool: update_node {"node_id": "4", "inputs": {"width": 1024, "height": 1024}}
→ tool: save_workflow {"name": "sdxl_txt2img"}

Workflow saved to workflows/sdxl_txt2img_20250508_120000.json.
```

```
[You] Install the ComfyUI Impact Pack and add a face detailer after the VAE decode

→ tool: search_custom_nodes {"query": "Impact Pack"}
← [{"title":"ComfyUI Impact Pack","reference":"https://github.com/ltdrdata/ComfyUI-Impact-Pack",...}]
→ tool: install_node_by_title {"title": "ComfyUI Impact Pack"}
← {"success": true, ...}
→ tool: refresh_resources {}
→ tool: get_workflow_summary {}
→ tool: add_node {"class_type": "FaceDetailer", "inputs": {...}}
→ tool: save_workflow {"name": "sdxl_with_facedetailer"}
```

```
[You] Find and download the best SDXL checkpoint from CivitAI

→ tool: search_models_civitai {"query": "SDXL base", "model_type": "checkpoint", "limit": 5}
← [list of top models with download URLs]
→ tool: download_model {"url": "...", "category": "checkpoints", "filename": "juggernautXL.safetensors"}
← {"success": true, "size_mb": 6800, ...}
→ tool: refresh_resources {}
```
