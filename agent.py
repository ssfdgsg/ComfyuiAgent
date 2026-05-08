"""
ComfyUI Agent — main entry point.

Architecture
============
1. A single WorkflowManager holds the in-memory workflow.
2. StateManager documents are refreshed after mutating operations.
3. Claude (claude-sonnet-4-6) drives the agent loop via tool use.
4. Tool descriptions keep the JSON payload small: summaries go to Claude,
   full node details only on request.

Usage
=====
  python agent.py "create a SDXL txt2img workflow with ControlNet"
  python agent.py   # interactive REPL
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import traceback
from typing import Any

import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ANTHROPIC_API_KEY, AGENT_MODEL, MAX_TOKENS, MAX_TOOL_ROUNDS, WORKFLOWS_DIR
from tools.comfyui_api import get_client
from tools.workflow import WorkflowManager, PRESET_BUILDERS
from tools.models import (
    scan_local_models, get_model_summary,
    search_huggingface, search_civitai, get_hf_model_files,
    download_model, download_model_background, get_download_progress,
    get_comfyui_model_list,
)
from tools.nodes import (
    list_installed_packages, get_installed_summary,
    fetch_node_registry, search_nodes,
    install_custom_node, install_node_by_title,
    update_custom_node, uninstall_custom_node,
)
from tools.state import (
    refresh_resources_doc, read_resources_doc,
    update_workflow_state, read_workflow_state,
    log_operation, history_summary, get_agent_context,
)

console = Console()

# ============================================================== tool registry
TOOLS: list[dict] = [
    # -------------------------------------------------- workflow: inspection
    {
        "name": "get_workflow_summary",
        "description": (
            "Return a compact text summary of the current in-memory workflow. "
            "Always use this first to understand the workflow before making changes. "
            "Returns node IDs, class types, connections, and key parameters."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_node_detail",
        "description": "Return the full JSON definition of a single node by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "The node ID string (e.g. '5')"}
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_nodes_by_type",
        "description": "Return all nodes of a given class_type (e.g. 'KSampler').",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_type": {"type": "string"}
            },
            "required": ["class_type"],
        },
    },
    # -------------------------------------------------- workflow: mutation
    {
        "name": "add_node",
        "description": (
            "Add a single node to the workflow. "
            "Input values can be scalars or link references [\"source_node_id\", output_slot_index]."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "class_type": {"type": "string", "description": "ComfyUI node class name"},
                "inputs": {
                    "type": "object",
                    "description": "Key-value inputs. Link format: [\"src_node_id\", slot_int]",
                },
                "title": {"type": "string", "description": "Display title (optional)"},
                "node_id": {"type": "string", "description": "Force a specific ID (optional)"},
            },
            "required": ["class_type"],
        },
    },
    {
        "name": "remove_node",
        "description": "Remove a node and all links pointing to it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"}
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "update_node",
        "description": "Update (merge) the inputs/parameters of an existing node.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "inputs": {"type": "object", "description": "Fields to update"},
            },
            "required": ["node_id", "inputs"],
        },
    },
    {
        "name": "create_link",
        "description": "Connect an output slot of one node to an input of another.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_node_id": {"type": "string"},
                "from_slot": {"type": "integer", "description": "Output slot index (0-based)"},
                "to_node_id": {"type": "string"},
                "to_input_name": {"type": "string", "description": "Target input field name"},
            },
            "required": ["from_node_id", "from_slot", "to_node_id", "to_input_name"],
        },
    },
    {
        "name": "remove_link",
        "description": "Disconnect the link feeding a specific input of a node.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "input_name": {"type": "string"},
            },
            "required": ["node_id", "input_name"],
        },
    },
    {
        "name": "apply_workflow_patch",
        "description": (
            "Apply multiple workflow changes in one call (batch operation). "
            "Accepts 'add', 'remove', 'update', 'link', 'unlink' lists. "
            "Prefer this over individual calls when making 3+ changes at once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "object",
                    "description": textwrap.dedent("""
                        {
                          "add":    [{"class_type":"..","inputs":{},"title":"","node_id":""}],
                          "remove": ["node_id"],
                          "update": [{"node_id":"..","inputs":{}}],
                          "link":   [{"from":"id","from_slot":0,"to":"id","to_input":"name"}],
                          "unlink": [{"node_id":"id","input_name":"name"}]
                        }
                    """).strip(),
                }
            },
            "required": ["patch"],
        },
    },
    # -------------------------------------------------- workflow: lifecycle
    {
        "name": "load_workflow",
        "description": "Load a workflow JSON file into memory (replaces current workflow).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "load_workflow_preset",
        "description": (
            f"Load a built-in workflow preset. "
            f"Available presets: {list(PRESET_BUILDERS.keys())}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": list(PRESET_BUILDERS.keys()),
                }
            },
            "required": ["preset"],
        },
    },
    {
        "name": "save_workflow",
        "description": "Save the current workflow to a JSON file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Filename prefix (no extension)"},
                "path": {"type": "string", "description": "Full path override (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "queue_workflow",
        "description": (
            "Send the current workflow to ComfyUI for execution. "
            "Returns the prompt_id for status tracking."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_execution_status",
        "description": "Check the ComfyUI queue and recent history.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ------------------------------------------ ComfyUI node type discovery
    {
        "name": "list_available_node_types",
        "description": (
            "Return all ComfyUI node class names registered in the running instance. "
            "Use this to discover valid class_type values before adding nodes."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_node_type_info",
        "description": "Get the input/output definition for a specific ComfyUI node class.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_type": {"type": "string"}
            },
            "required": ["class_type"],
        },
    },
    # --------------------------------------------------- model management
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
                "model_type": {
                    "type": "string",
                    "description": "HF pipeline_tag filter (e.g. 'text-to-image')",
                },
                "limit": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_models_civitai",
        "description": "Search CivitAI for models (checkpoints, LoRAs, ControlNets, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "model_type": {
                    "type": "string",
                    "description": "One of: checkpoint, lora, lycoris, textual_inversion, hypernetwork, controlnet, vae, upscaler",
                },
                "limit": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_huggingface_model_files",
        "description": "List downloadable files for a HuggingFace repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "e.g. 'stabilityai/stable-diffusion-xl-base-1.0'"}
            },
            "required": ["repo_id"],
        },
    },
    {
        "name": "search_comfyui_model_list",
        "description": "Search the official ComfyUI Manager model registry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "download_model",
        "description": (
            "Download a model file to the correct ComfyUI models sub-directory. "
            "The download runs synchronously and shows progress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Direct download URL"},
                "category": {
                    "type": "string",
                    "description": "Model category: checkpoints, vae, loras, controlnet, embeddings, upscale_models, clip, unet, clip_vision, ipadapter, ...",
                },
                "filename": {"type": "string", "description": "Override filename (optional)"},
            },
            "required": ["url", "category"],
        },
    },
    {
        "name": "download_model_background",
        "description": (
            "Start a model download in the background (non-blocking). "
            "Returns immediately. Use get_download_status to track progress."
        ),
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
        "description": "Check the progress of background model downloads.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ------------------------------------------------ custom node management
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
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "install_custom_node",
        "description": "Install a custom node package from a git URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "git_url": {"type": "string", "description": "GitHub/GitLab URL of the custom node repo"},
                "pip_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional pip packages to install",
                },
            },
            "required": ["git_url"],
        },
    },
    {
        "name": "install_node_by_title",
        "description": "Install a custom node by its title from the ComfyUI Manager registry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact node title from registry"}
            },
            "required": ["title"],
        },
    },
    {
        "name": "update_custom_node",
        "description": "Pull latest updates for an installed custom node.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Package directory name"}
            },
            "required": ["name"],
        },
    },
    # ---------------------------------------------------- state / docs
    {
        "name": "refresh_resources",
        "description": "Re-scan models and custom nodes; rewrite resources.md. Call this after installing anything.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_resources",
        "description": "Read the current resources.md document (models + custom nodes).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_workflow_state",
        "description": "Read the persisted workflow state document.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_operation_history",
        "description": "Read recent operation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 20}
            },
            "required": [],
        },
    },
    {
        "name": "list_saved_workflows",
        "description": "List workflow JSON files saved in the workflows/ directory.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ================================================================ tool router
class Agent:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.wf = WorkflowManager()
        self._current_wf_file: str = ""

    # ----------------------------------------------------- tool dispatch
    def _dispatch(self, name: str, inp: dict) -> Any:
        try:
            return self._run_tool(name, inp)
        except Exception as e:
            tb = traceback.format_exc()
            return {"error": str(e), "traceback": tb}

    def _run_tool(self, name: str, inp: dict) -> Any:
        # ---- workflow inspection ----
        if name == "get_workflow_summary":
            return self.wf.get_summary()

        if name == "get_node_detail":
            node = self.wf.get_node(inp["node_id"])
            return node if node else {"error": f"Node {inp['node_id']} not found"}

        if name == "get_nodes_by_type":
            return self.wf.get_nodes_by_type(inp["class_type"])

        # ---- workflow mutation ----
        if name == "add_node":
            nid = self.wf.add_node(
                class_type=inp["class_type"],
                inputs=inp.get("inputs"),
                title=inp.get("title", ""),
                node_id=inp.get("node_id"),
            )
            self._save_state("add_node", {"class_type": inp["class_type"], "node_id": nid})
            return {"node_id": nid, "class_type": inp["class_type"]}

        if name == "remove_node":
            ok = self.wf.remove_node(inp["node_id"])
            self._save_state("remove_node", {"node_id": inp["node_id"], "success": ok})
            return {"success": ok}

        if name == "update_node":
            ok = self.wf.update_node_inputs(inp["node_id"], inp["inputs"])
            self._save_state("update_node", {"node_id": inp["node_id"]})
            return {"success": ok}

        if name == "create_link":
            ok = self.wf.create_link(
                inp["from_node_id"], inp["from_slot"],
                inp["to_node_id"], inp["to_input_name"]
            )
            self._save_state("create_link", inp)
            return {"success": ok}

        if name == "remove_link":
            ok = self.wf.remove_link(inp["node_id"], inp["input_name"])
            self._save_state("remove_link", inp)
            return {"success": ok}

        if name == "apply_workflow_patch":
            log = self.wf.apply_patch(inp["patch"])
            self._save_state("apply_patch", {"changes": len(log)})
            return {"log": log}

        # ---- workflow lifecycle ----
        if name == "load_workflow":
            path = inp["path"]
            if not os.path.isabs(path):
                path = os.path.join(WORKFLOWS_DIR, path)
            self.wf = WorkflowManager.from_file(path)
            self._current_wf_file = path
            self._save_state("load_workflow", {"path": path, "nodes": self.wf.node_count()})
            return {"loaded": path, "node_count": self.wf.node_count()}

        if name == "load_workflow_preset":
            builder = PRESET_BUILDERS[inp["preset"]]
            self.wf = builder()
            self._current_wf_file = ""
            self._save_state("load_preset", {"preset": inp["preset"]})
            return {"preset": inp["preset"], "node_count": self.wf.node_count(),
                    "summary": self.wf.get_summary()}

        if name == "save_workflow":
            path = inp.get("path")
            saved = self.wf.save(path=path, name=inp.get("name"))
            self._current_wf_file = saved
            self._save_state("save_workflow", {"path": saved})
            return {"saved": saved}

        if name == "queue_workflow":
            c = get_client()
            if not c.is_alive():
                return {"error": "ComfyUI is not reachable at " + c.base}
            result = c.queue_prompt(self.wf.to_api_format())
            self._save_state("queue_workflow", result)
            return result

        if name == "get_execution_status":
            c = get_client()
            if not c.is_alive():
                return {"comfyui": "offline"}
            queue = c.get_queue()
            history = c.get_history(max_items=5)
            return {"queue": queue, "recent_history_ids": list(history.keys())}

        # ---- ComfyUI node type discovery ----
        if name == "list_available_node_types":
            c = get_client()
            if not c.is_alive():
                return {"error": "ComfyUI offline — cannot query node types"}
            types = c.list_available_node_types()
            return {"count": len(types), "types": types}

        if name == "get_node_type_info":
            c = get_client()
            if not c.is_alive():
                return {"error": "ComfyUI offline"}
            return c.get_object_info_node(inp["class_type"])

        # ---- model management ----
        if name == "list_local_models":
            return scan_local_models()

        if name == "search_models_huggingface":
            return search_huggingface(
                inp["query"],
                model_type=inp.get("model_type", ""),
                limit=inp.get("limit", 8),
            )

        if name == "search_models_civitai":
            return search_civitai(
                inp["query"],
                model_type=inp.get("model_type", ""),
                limit=inp.get("limit", 8),
            )

        if name == "get_huggingface_model_files":
            return get_hf_model_files(inp["repo_id"])

        if name == "search_comfyui_model_list":
            q = inp["query"].lower()
            models = get_comfyui_model_list()
            results = [
                m for m in models
                if q in str(m.get("name", "")).lower()
                or q in str(m.get("description", "")).lower()
            ]
            return results[:15]

        if name == "download_model":
            result = download_model(
                url=inp["url"],
                category=inp["category"],
                filename=inp.get("filename"),
            )
            if result["success"]:
                refresh_resources_doc()
                self._save_state("download_model", {
                    "category": inp["category"],
                    "filename": inp.get("filename", ""),
                    "size_mb": result["size_mb"],
                })
            return result

        if name == "download_model_background":
            key = download_model_background(
                url=inp["url"],
                category=inp["category"],
                filename=inp.get("filename"),
            )
            return {"tracking_key": key, "status": "started"}

        if name == "get_download_status":
            return get_download_progress()

        # ---- custom nodes ----
        if name == "list_installed_nodes":
            return list_installed_packages()

        if name == "search_custom_nodes":
            return search_nodes(inp["query"], limit=inp.get("limit", 10))

        if name == "install_custom_node":
            result = install_custom_node(inp["git_url"], inp.get("pip_packages"))
            if result["success"]:
                refresh_resources_doc()
                self._save_state("install_node", {"git_url": inp["git_url"], "name": result["name"]})
            return result

        if name == "install_node_by_title":
            result = install_node_by_title(inp["title"])
            if result.get("success"):
                refresh_resources_doc()
                self._save_state("install_node", {"title": inp["title"]})
            return result

        if name == "update_custom_node":
            return update_custom_node(inp["name"])

        # ---- state / docs ----
        if name == "refresh_resources":
            content = refresh_resources_doc()
            return {"refreshed": True, "preview": content[:500]}

        if name == "read_resources":
            return read_resources_doc()

        if name == "read_workflow_state":
            return read_workflow_state()

        if name == "read_operation_history":
            return history_summary(n=inp.get("n", 20))

        if name == "list_saved_workflows":
            os.makedirs(WORKFLOWS_DIR, exist_ok=True)
            files = [
                f for f in os.listdir(WORKFLOWS_DIR) if f.endswith(".json")
            ]
            return {"files": sorted(files)}

        return {"error": f"Unknown tool: {name}"}

    def _save_state(self, op: str, details: dict):
        """Persist state doc + log after a mutating operation."""
        log_operation(op, details)
        if self.wf.node_count() > 0:
            update_workflow_state(
                summary=self.wf.get_summary(),
                workflow_file=self._current_wf_file,
            )

    # --------------------------------------------------------- agent loop
    def run(self, user_message: str) -> str:
        """Run one user turn through the full tool-use loop."""
        # Build system prompt with current state context
        context = get_agent_context()
        system = textwrap.dedent(f"""
            You are a ComfyUI automation agent running inside a yanwk/comfyui-boot container.

            Your capabilities:
            - Build and modify ComfyUI workflows (add/remove nodes, create links)
            - Search and install models from HuggingFace and CivitAI
            - Search and install custom node packages from the ComfyUI Manager registry
            - Execute workflows and monitor results

            Guidelines:
            1. Always call get_workflow_summary before modifying the workflow.
            2. When creating a workflow from scratch, prefer apply_workflow_patch to build
               multiple nodes at once rather than repeated add_node calls.
            3. Before adding a node, verify its class_type exists via list_available_node_types
               if ComfyUI is running; otherwise rely on built-in knowledge.
            4. For long workflows, use get_nodes_by_type or get_node_detail to fetch only
               the parts you need — never dump the full JSON unless explicitly asked.
            5. After installing models or nodes, call refresh_resources so the state doc
               reflects the change.
            6. When a workflow is complete, always call save_workflow.

            {context}
        """).strip()

        messages: list[dict] = [{"role": "user", "content": user_message}]
        console.print(Panel(f"[bold cyan]User:[/] {user_message}", expand=False))

        for _round in range(MAX_TOOL_ROUNDS):
            response = self.client.messages.create(
                model=AGENT_MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            # Collect all text + tool use blocks
            tool_uses = []
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if text_parts:
                combined_text = "\n".join(text_parts)
                console.print(Markdown(combined_text))

            if response.stop_reason == "end_turn" or not tool_uses:
                return "\n".join(text_parts)

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tools in this turn (possibly multiple)
            tool_results = []
            for tu in tool_uses:
                console.print(f"  [dim]→ tool: [yellow]{tu.name}[/] {json.dumps(tu.input)[:120]}[/]")
                result = self._dispatch(tu.name, tu.input)
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                if len(result_str) > 6000:
                    result_str = result_str[:5900] + "\n... (truncated)"
                console.print(f"  [dim]← {result_str[:200]}[/]")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        return "Max tool rounds reached."

    def repl(self):
        """Interactive REPL mode."""
        console.print(Panel(
            "[bold green]ComfyUI Agent[/]\n"
            "Type your instruction, or 'exit' to quit.\n"
            f"Model: {AGENT_MODEL}  |  ComfyUI: {get_client().base}",
            title="Agent Ready"
        ))
        while True:
            try:
                user_input = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Goodbye.[/]")
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                console.print("[yellow]Goodbye.[/]")
                break
            self.run(user_input)


# ================================================================ entrypoint
def main():
    agent = Agent()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        agent.run(task)
    else:
        agent.repl()


if __name__ == "__main__":
    main()
