"""
ComfyUI Agent — core logic and CLI entry point.

AgentCore  — holds in-memory workflow + all tool implementations (sync).
Agent      — wraps AgentCore with an LLM provider loop.
             .run_stream() → async generator of events (for web).
             .run_sync()   → blocking string result (for CLI).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import traceback
from typing import Any, AsyncGenerator

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MAX_TOOL_ROUNDS, WORKFLOWS_DIR,
    GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_PROVIDER,
)
from tools.comfyui_api import get_client
from tools.workflow import WorkflowManager, PRESET_BUILDERS
from tools.models import (
    scan_local_models, get_model_summary,
    search_huggingface, search_civitai, get_hf_model_files,
    download_model, download_model_background, get_download_progress,
    get_comfyui_model_list,
    search_modelscope, download_from_modelscope,
)
from tools.nodes import (
    list_installed_packages, get_installed_summary,
    search_nodes, install_custom_node, install_node_by_title,
    update_custom_node,
)
from tools.state import (
    refresh_resources_doc, read_resources_doc,
    update_workflow_state, read_workflow_state,
    log_operation, history_summary, get_agent_context,
)
from llm.base import MediaFile, TextChunk, ToolCallEvent, DoneEvent

# ============================================================= tool definitions
TOOLS: list[dict] = [
    # ─── workflow inspection ───
    {
        "name": "get_workflow_summary",
        "description": "Return compact summary of the current workflow. Call this before any modification.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_node_detail",
        "description": "Return full JSON of a single node by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "get_nodes_by_type",
        "description": "Return all nodes of a given class_type (e.g. 'KSampler').",
        "input_schema": {
            "type": "object",
            "properties": {"class_type": {"type": "string"}},
            "required": ["class_type"],
        },
    },
    # ─── workflow mutation ───
    {
        "name": "apply_workflow_patch",
        "description": (
            "Apply multiple workflow changes atomically. "
            "Accepts 'add', 'remove', 'update', 'link', 'unlink' lists. "
            "Prefer over individual calls for 3+ changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "object",
                    "description": (
                        '{"add":[{"class_type":"..","inputs":{},"title":"","node_id":""}],'
                        '"remove":["id"],'
                        '"update":[{"node_id":"..","inputs":{}}],'
                        '"link":[{"from":"id","from_slot":0,"to":"id","to_input":"name"}],'
                        '"unlink":[{"node_id":"id","input_name":"name"}]}'
                    ),
                }
            },
            "required": ["patch"],
        },
    },
    {
        "name": "add_node",
        "description": "Add a single node. Input values can be scalars or [src_id, slot] links.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_type": {"type": "string"},
                "inputs": {"type": "object"},
                "title": {"type": "string"},
                "node_id": {"type": "string"},
            },
            "required": ["class_type"],
        },
    },
    {
        "name": "remove_node",
        "description": "Remove a node and all links pointing to it.",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "update_node",
        "description": "Merge updates into a node's inputs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "inputs": {"type": "object"},
            },
            "required": ["node_id", "inputs"],
        },
    },
    {
        "name": "create_link",
        "description": "Connect output slot of one node to input of another.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_node_id": {"type": "string"},
                "from_slot": {"type": "integer"},
                "to_node_id": {"type": "string"},
                "to_input_name": {"type": "string"},
            },
            "required": ["from_node_id", "from_slot", "to_node_id", "to_input_name"],
        },
    },
    {
        "name": "remove_link",
        "description": "Disconnect the link feeding a specific node input.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "input_name": {"type": "string"},
            },
            "required": ["node_id", "input_name"],
        },
    },
    # ─── workflow lifecycle ───
    {
        "name": "load_workflow_preset",
        "description": f"Load a built-in preset. Available: {list(PRESET_BUILDERS.keys())}",
        "input_schema": {
            "type": "object",
            "properties": {"preset": {"type": "string", "enum": list(PRESET_BUILDERS.keys())}},
            "required": ["preset"],
        },
    },
    {
        "name": "load_workflow",
        "description": "Load a workflow JSON file from the workflows/ directory or absolute path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "save_workflow",
        "description": "Save current workflow to a JSON file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "queue_workflow",
        "description": "Send the current workflow to ComfyUI for execution.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_execution_status",
        "description": "Check ComfyUI queue and recent history.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_available_node_types",
        "description": "Return all node class names registered in the running ComfyUI instance.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_node_type_info",
        "description": "Get input/output definition for a specific ComfyUI node class.",
        "input_schema": {
            "type": "object",
            "properties": {"class_type": {"type": "string"}},
            "required": ["class_type"],
        },
    },
    # ─── model management ───
    {
        "name": "list_local_models",
        "description": "List all model files installed in the ComfyUI models directory.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_models_huggingface",
        "description": "Search HuggingFace Hub for models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "model_type": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_models_civitai",
        "description": "Search CivitAI for checkpoints, LoRAs, ControlNets, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "model_type": {"type": "string", "description": "checkpoint|lora|controlnet|vae|upscaler"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_models_modelscope",
        "description": "Search ModelScope (魔搭) for models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_huggingface_model_files",
        "description": "List downloadable files for a HuggingFace repo.",
        "input_schema": {
            "type": "object",
            "properties": {"repo_id": {"type": "string"}},
            "required": ["repo_id"],
        },
    },
    {
        "name": "download_model",
        "description": "Download a model from a direct URL to the correct ComfyUI directory (streaming).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "category": {"type": "string", "description": "checkpoints|vae|loras|controlnet|embeddings|..."},
                "filename": {"type": "string"},
            },
            "required": ["url", "category"],
        },
    },
    {
        "name": "download_from_modelscope",
        "description": "Download a model from ModelScope (魔搭) using the modelscope SDK.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "e.g. 'AI-ModelScope/stable-diffusion-xl-base-1.0'"},
                "category": {"type": "string"},
                "file_pattern": {"type": "string", "description": "e.g. '*.safetensors'"},
            },
            "required": ["model_id", "category"],
        },
    },
    {
        "name": "download_model_background",
        "description": "Start a URL model download in the background. Use get_download_status to track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "category": {"type": "string"},
                "filename": {"type": "string"},
            },
            "required": ["url", "category"],
        },
    },
    {
        "name": "get_download_status",
        "description": "Check progress of background model downloads.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ─── custom nodes ───
    {
        "name": "list_installed_nodes",
        "description": "List all installed custom node packages.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_custom_nodes",
        "description": "Search the ComfyUI Manager custom node registry.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "install_custom_node",
        "description": "Install a custom node package from a git URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "git_url": {"type": "string"},
                "pip_packages": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["git_url"],
        },
    },
    {
        "name": "install_node_by_title",
        "description": "Install a custom node by its title from the ComfyUI Manager registry.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
    {
        "name": "update_custom_node",
        "description": "Pull latest updates for an installed custom node package.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    # ─── state / docs ───
    {
        "name": "refresh_resources",
        "description": "Re-scan models and custom nodes; rewrite resources.md. Call after installing anything.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_resources",
        "description": "Read the current resources.md (models + custom nodes).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_saved_workflows",
        "description": "List saved workflow JSON files.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a ComfyUI automation agent running inside a yanwk/comfyui-boot container.

    Capabilities:
    - Build and modify ComfyUI workflows (add/remove nodes, links, parameters)
    - Search and download models from HuggingFace, CivitAI, or ModelScope (魔搭)
    - Search and install custom node packages from the ComfyUI Manager registry
    - Execute workflows and monitor results

    Rules:
    1. Always call get_workflow_summary before modifying the workflow.
    2. Use apply_workflow_patch for 3+ changes in a single call.
    3. After installing models or nodes, call refresh_resources.
    4. Always call save_workflow when the workflow is complete.
    5. Verify node class_type with list_available_node_types when ComfyUI is reachable.
""").strip()


# ============================================================= AgentCore
class AgentCore:
    """Holds workflow state and implements all tool dispatch (synchronous)."""

    def __init__(self):
        self.wf = WorkflowManager()
        self._current_wf_file: str = ""

    # ---------------------------------------------------------------- dispatch
    def dispatch_tool(self, name: str, inp: dict) -> Any:
        try:
            return self._run_tool(name, inp)
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _run_tool(self, name: str, inp: dict) -> Any:
        # ── workflow inspection ──
        if name == "get_workflow_summary":
            return self.wf.get_summary()
        if name == "get_node_detail":
            node = self.wf.get_node(inp["node_id"])
            return node or {"error": f"Node {inp['node_id']} not found"}
        if name == "get_nodes_by_type":
            return self.wf.get_nodes_by_type(inp["class_type"])

        # ── workflow mutation ──
        if name == "apply_workflow_patch":
            log = self.wf.apply_patch(inp["patch"])
            self._save_state("apply_patch", {"changes": len(log)})
            return {"log": log}
        if name == "add_node":
            nid = self.wf.add_node(
                class_type=inp["class_type"],
                inputs=inp.get("inputs"),
                title=inp.get("title", ""),
                node_id=inp.get("node_id"),
            )
            self._save_state("add_node", {"class_type": inp["class_type"], "node_id": nid})
            return {"node_id": nid}
        if name == "remove_node":
            ok = self.wf.remove_node(inp["node_id"])
            self._save_state("remove_node", {"node_id": inp["node_id"]})
            return {"success": ok}
        if name == "update_node":
            ok = self.wf.update_node_inputs(inp["node_id"], inp["inputs"])
            self._save_state("update_node", {"node_id": inp["node_id"]})
            return {"success": ok}
        if name == "create_link":
            ok = self.wf.create_link(inp["from_node_id"], inp["from_slot"], inp["to_node_id"], inp["to_input_name"])
            self._save_state("create_link", inp)
            return {"success": ok}
        if name == "remove_link":
            ok = self.wf.remove_link(inp["node_id"], inp["input_name"])
            return {"success": ok}

        # ── workflow lifecycle ──
        if name == "load_workflow_preset":
            builder = PRESET_BUILDERS[inp["preset"]]
            self.wf = builder()
            self._current_wf_file = ""
            self._save_state("load_preset", {"preset": inp["preset"]})
            return {"node_count": self.wf.node_count(), "summary": self.wf.get_summary()}
        if name == "load_workflow":
            path = inp["path"]
            if not os.path.isabs(path):
                path = os.path.join(WORKFLOWS_DIR, path)
            self.wf = WorkflowManager.from_file(path)
            self._current_wf_file = path
            self._save_state("load_workflow", {"path": path})
            return {"loaded": path, "node_count": self.wf.node_count()}
        if name == "save_workflow":
            path = inp.get("path")
            saved = self.wf.save(path=path, name=inp.get("name"))
            self._current_wf_file = saved
            self._save_state("save_workflow", {"path": saved})
            return {"saved": saved}
        if name == "queue_workflow":
            c = get_client()
            if not c.is_alive():
                return {"error": "ComfyUI not reachable at " + c.base}
            result = c.queue_prompt(self.wf.to_api_format())
            self._save_state("queue_workflow", result)
            return result
        if name == "get_execution_status":
            c = get_client()
            if not c.is_alive():
                return {"comfyui": "offline"}
            return {"queue": c.get_queue(), "recent_ids": list(c.get_history(5).keys())}
        if name == "list_available_node_types":
            c = get_client()
            if not c.is_alive():
                return {"error": "ComfyUI offline"}
            types = c.list_available_node_types()
            return {"count": len(types), "types": types}
        if name == "get_node_type_info":
            c = get_client()
            if not c.is_alive():
                return {"error": "ComfyUI offline"}
            return c.get_object_info_node(inp["class_type"])

        # ── models ──
        if name == "list_local_models":
            return scan_local_models()
        if name == "search_models_huggingface":
            return search_huggingface(inp["query"], model_type=inp.get("model_type", ""), limit=inp.get("limit", 8))
        if name == "search_models_civitai":
            return search_civitai(inp["query"], model_type=inp.get("model_type", ""), limit=inp.get("limit", 8))
        if name == "search_models_modelscope":
            return search_modelscope(inp["query"], limit=inp.get("limit", 10))
        if name == "get_huggingface_model_files":
            return get_hf_model_files(inp["repo_id"])
        if name == "download_model":
            result = download_model(url=inp["url"], category=inp["category"], filename=inp.get("filename"))
            if result["success"]:
                refresh_resources_doc()
                self._save_state("download_model", {"category": inp["category"], "size_mb": result["size_mb"]})
            return result
        if name == "download_from_modelscope":
            result = download_from_modelscope(
                model_id=inp["model_id"], category=inp["category"],
                file_pattern=inp.get("file_pattern", "*.safetensors"),
            )
            if result["success"]:
                refresh_resources_doc()
                self._save_state("download_modelscope", {"model_id": inp["model_id"]})
            return result
        if name == "download_model_background":
            key = download_model_background(url=inp["url"], category=inp["category"], filename=inp.get("filename"))
            return {"tracking_key": key, "status": "started"}
        if name == "get_download_status":
            return get_download_progress()

        # ── custom nodes ──
        if name == "list_installed_nodes":
            return list_installed_packages()
        if name == "search_custom_nodes":
            return search_nodes(inp["query"], limit=inp.get("limit", 10))
        if name == "install_custom_node":
            result = install_custom_node(inp["git_url"], inp.get("pip_packages"))
            if result["success"]:
                refresh_resources_doc()
                self._save_state("install_node", {"git_url": inp["git_url"]})
            return result
        if name == "install_node_by_title":
            result = install_node_by_title(inp["title"])
            if result.get("success"):
                refresh_resources_doc()
                self._save_state("install_node", {"title": inp["title"]})
            return result
        if name == "update_custom_node":
            return update_custom_node(inp["name"])

        # ── state ──
        if name == "refresh_resources":
            content = refresh_resources_doc()
            return {"refreshed": True, "preview": content[:400]}
        if name == "read_resources":
            return read_resources_doc()
        if name == "list_saved_workflows":
            os.makedirs(WORKFLOWS_DIR, exist_ok=True)
            return {"files": sorted(f for f in os.listdir(WORKFLOWS_DIR) if f.endswith(".json"))}

        return {"error": f"Unknown tool: {name}"}

    def _save_state(self, op: str, details: dict):
        log_operation(op, details)
        if self.wf.node_count() > 0:
            update_workflow_state(
                summary=self.wf.get_summary(),
                workflow_file=self._current_wf_file,
            )

    def workflow_nodes_json(self) -> list[dict]:
        """Return nodes as a list suitable for the web UI."""
        nodes = []
        for nid, node in self.wf._nodes.items():
            nodes.append({
                "id": nid,
                "class_type": node.get("class_type", ""),
                "title": node.get("_meta", {}).get("title", ""),
                "inputs": node.get("inputs", {}),
            })
        return sorted(nodes, key=lambda x: int(x["id"]) if x["id"].isdigit() else 0)


# ============================================================= Agent (LLM loop)
class Agent:
    def __init__(
        self,
        provider: str = "",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ):
        from llm.provider import create_provider
        self.core = AgentCore()
        self._provider_name = provider or DEFAULT_PROVIDER
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._llm = self._make_provider()

    def _make_provider(self):
        from llm.provider import create_provider
        return create_provider(
            provider=self._provider_name,
            api_key=self._api_key,
            model=self._model,
            base_url=self._base_url,
        )

    def update_settings(self, provider: str = "", api_key: str = "", model: str = "", base_url: str = ""):
        """Hot-reload provider settings (called from web /api/settings)."""
        if provider:
            self._provider_name = provider
        if api_key:
            self._api_key = api_key
        if model:
            self._model = model
        if base_url:
            self._base_url = base_url
        self._llm = self._make_provider()

    def _build_system(self) -> str:
        context = get_agent_context()
        return SYSTEM_PROMPT + "\n\n" + context

    async def run_stream(
        self,
        user_message: str,
        media_files: list[MediaFile] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields dicts for the WebSocket:
          {"type": "token",          "content": "..."}
          {"type": "tool_call",      "name": "...", "input": {...}}
          {"type": "tool_result",    "name": "...", "result": {...}}
          {"type": "workflow_update","nodes": [...], "summary": "..."}
          {"type": "done",           "content": "..."}
        """
        messages: list[dict] = [{"role": "user", "content": user_message}]
        system = self._build_system()

        for _round in range(MAX_TOOL_ROUNDS):
            tool_calls_this_turn: list[ToolCallEvent] = []
            full_text = ""

            async for event in self._llm.generate_stream(
                messages=messages,
                tools=TOOLS,
                system=system,
                media_files=media_files if _round == 0 else None,
            ):
                if isinstance(event, TextChunk):
                    full_text += event.content
                    yield {"type": "token", "content": event.content}
                elif isinstance(event, ToolCallEvent):
                    tool_calls_this_turn.append(event)
                    yield {"type": "tool_call", "id": event.id, "name": event.name, "input": event.input}
                elif isinstance(event, DoneEvent):
                    full_text = event.content

            if not tool_calls_this_turn:
                # Pure text response — we're done
                yield {"type": "done", "content": full_text}
                return

            # Append assistant turn (text + tool calls)
            assistant_content: list[dict] = []
            if full_text:
                assistant_content.append({"type": "text", "text": full_text})
            for tc in tool_calls_this_turn:
                assistant_content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tools and append results
            for tc in tool_calls_this_turn:
                result = await asyncio.to_thread(self.core.dispatch_tool, tc.name, tc.input)
                yield {"type": "tool_result", "name": tc.name, "result": result}
                messages.append(
                    self.core.dispatch_tool.__self__.__class__.dispatch_tool  # type: ignore
                    if False else {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result,
                    }
                )

            # After any workflow mutation, emit current state
            if self.core.wf.node_count() > 0:
                yield {
                    "type": "workflow_update",
                    "nodes": self.core.workflow_nodes_json(),
                    "summary": self.core.wf.get_summary(),
                }

        yield {"type": "done", "content": "Max iterations reached."}

    def run_sync(self, user_message: str) -> str:
        """Blocking CLI version."""
        return asyncio.run(self._collect(user_message))

    async def _collect(self, user_message: str) -> str:
        parts = []
        async for event in self.run_stream(user_message):
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
                parts.append(event["content"])
            elif event["type"] == "tool_call":
                print(f"\n[tool] {event['name']} {json.dumps(event['input'])[:80]}", flush=True)
            elif event["type"] == "tool_result":
                r = json.dumps(event["result"])[:120]
                print(f"[result] {r}", flush=True)
            elif event["type"] == "done":
                print()
                return event["content"]
        return "".join(parts)


# ============================================================= CLI
def main():
    import sys as _sys
    if len(_sys.argv) > 1:
        task = " ".join(_sys.argv[1:])
        agent = Agent()
        agent.run_sync(task)
    else:
        agent = Agent()
        print(f"ComfyUI Agent REPL (provider: {agent._provider_name}) — type 'exit' to quit")
        while True:
            try:
                msg = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg or msg.lower() in ("exit", "quit"):
                break
            agent.run_sync(msg)


if __name__ == "__main__":
    main()
