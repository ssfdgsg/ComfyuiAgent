"""
Workflow manager with smart context minimization.

ComfyUI workflow JSON can have hundreds of nodes. Rather than dumping the
full JSON into the LLM context, WorkflowManager maintains an in-memory
representation and exposes:

  - get_summary()          → compact text snapshot (always small)
  - get_node_detail(id)    → full dict for one node (on demand)
  - get_nodes_by_type(t)   → subset for a node class
  - apply_patch(patch)     → incremental update without reloading

The "API format" that ComfyUI /prompt accepts uses node IDs as string keys.
The "UI format" (exported from the browser) has a different structure with
a "nodes" list and "links" list. This manager works with API format natively
and can import/export UI format.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import WORKFLOWS_DIR


class WorkflowManager:
    """
    In-memory ComfyUI workflow (API format).

    API format structure:
    {
        "<node_id>": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "model": ["4", 0],   # ["source_node_id", output_slot]
                ...
            },
            "_meta": {"title": "KSampler"}   # optional
        },
        ...
    }
    """

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._next_id: int = 1
        self._dirty: bool = False

    # ----------------------------------------------------------------- load
    @classmethod
    def from_file(cls, path: str) -> "WorkflowManager":
        wf = cls()
        with open(path) as f:
            data = json.load(f)
        if "nodes" in data and "links" in data:
            wf._nodes = cls._ui_to_api(data)
        else:
            wf._nodes = {str(k): v for k, v in data.items()}
        wf._sync_next_id()
        return wf

    @classmethod
    def empty(cls) -> "WorkflowManager":
        return cls()

    def _sync_next_id(self):
        ids = [int(k) for k in self._nodes if k.isdigit()]
        self._next_id = max(ids, default=0) + 1

    # ---------------------------------------------------------- UI ↔ API fmt
    @staticmethod
    def _ui_to_api(ui: dict) -> dict[str, dict]:
        """Convert browser-exported UI format to API format."""
        api: dict[str, dict] = {}
        # Build link lookup: link_id -> [src_node_id, src_slot]
        link_map: dict[int, list] = {}
        for link in ui.get("links", []):
            # link = [link_id, src_node, src_slot, dst_node, dst_slot, type]
            link_map[link[0]] = [str(link[1]), link[2]]

        for node in ui.get("nodes", []):
            nid = str(node["id"])
            class_type = node.get("type", "")
            inputs: dict[str, Any] = {}

            # Widget values (non-linked inputs) are stored in order
            widget_values = list(node.get("widgets_values", []))
            widget_idx = 0

            # inputs list in UI format
            for inp in node.get("inputs", []):
                name = inp.get("name", "")
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    inputs[name] = link_map[link_id]
                # else: will be filled from widgets_values below

            # Fill remaining widget values for unlinked inputs
            # This is approximate — widget ordering depends on node definition
            remaining_names = [
                inp["name"]
                for inp in node.get("inputs", [])
                if inp.get("link") is None
            ]
            # Also include widget-only params not listed in inputs
            for name in remaining_names:
                if widget_idx < len(widget_values):
                    inputs[name] = widget_values[widget_idx]
                    widget_idx += 1

            entry: dict[str, Any] = {"class_type": class_type, "inputs": inputs}
            title = node.get("title")
            if title:
                entry["_meta"] = {"title": title}
            api[nid] = entry
        return api

    def to_api_format(self) -> dict:
        """Return a deep-copy safe for sending to /prompt."""
        return copy.deepcopy(self._nodes)

    # --------------------------------------------------------------- summary
    def get_summary(self) -> str:
        """
        Return a compact human-readable summary of the workflow.
        Stays small regardless of workflow size — safe to include in LLM context.
        """
        if not self._nodes:
            return "Workflow is empty (no nodes)."

        # Build connection map: node_id -> list of (input_name, src_node_id, src_slot)
        lines = [f"Workflow has {len(self._nodes)} nodes:\n"]
        for nid, node in sorted(self._nodes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            title = node.get("_meta", {}).get("title", "")
            label = f"[{nid}] {node['class_type']}"
            if title and title != node["class_type"]:
                label += f' "{title}"'
            inputs = node.get("inputs", {})
            links_in = [
                f"{k}←[{v[0]}:{v[1]}]"
                for k, v in inputs.items()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)
            ]
            scalars = {
                k: v
                for k, v in inputs.items()
                if not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], str))
            }
            parts = []
            if links_in:
                parts.append("links: " + ", ".join(links_in))
            if scalars:
                # Truncate long values
                scalar_strs = []
                for k, v in scalars.items():
                    sv = str(v)
                    if len(sv) > 60:
                        sv = sv[:57] + "..."
                    scalar_strs.append(f"{k}={sv}")
                parts.append("params: " + ", ".join(scalar_strs))
            lines.append(f"  {label}  " + "  |  ".join(parts))
        return "\n".join(lines)

    def get_graph_summary(self) -> str:
        """Return only the node IDs, types, and their connection graph."""
        if not self._nodes:
            return "Empty workflow."
        rows = []
        for nid, node in sorted(self._nodes.items()):
            ct = node["class_type"]
            inputs = node.get("inputs", {})
            deps = [
                f"{v[0]}"
                for v in inputs.values()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)
            ]
            dep_str = f" ← {{{', '.join(set(deps))}}}" if deps else ""
            rows.append(f"  [{nid}]{ct}{dep_str}")
        return "Node graph:\n" + "\n".join(rows)

    # --------------------------------------------------------- node access
    def get_node(self, node_id: str) -> dict | None:
        return copy.deepcopy(self._nodes.get(str(node_id)))

    def get_nodes_by_type(self, class_type: str) -> dict[str, dict]:
        return {
            k: copy.deepcopy(v)
            for k, v in self._nodes.items()
            if v.get("class_type") == class_type
        }

    def list_node_types(self) -> list[str]:
        return sorted(set(v["class_type"] for v in self._nodes.values()))

    def list_node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    # --------------------------------------------------------- node mutation
    def add_node(
        self,
        class_type: str,
        inputs: dict | None = None,
        title: str = "",
        node_id: str | None = None,
    ) -> str:
        """
        Add a node. Returns the assigned node_id.
        `inputs` values can be scalars or ["source_node_id", slot_index] links.
        """
        nid = str(node_id) if node_id else str(self._next_id)
        if nid in self._nodes:
            raise ValueError(f"Node ID {nid} already exists")
        entry: dict[str, Any] = {
            "class_type": class_type,
            "inputs": inputs or {},
        }
        if title:
            entry["_meta"] = {"title": title}
        self._nodes[nid] = entry
        self._next_id = max(self._next_id, int(nid) + 1) if nid.isdigit() else self._next_id + 1
        self._dirty = True
        return nid

    def remove_node(self, node_id: str) -> bool:
        nid = str(node_id)
        if nid not in self._nodes:
            return False
        del self._nodes[nid]
        # Remove any links pointing to deleted node
        for node in self._nodes.values():
            inputs = node.get("inputs", {})
            to_remove = [
                k for k, v in inputs.items()
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid
            ]
            for k in to_remove:
                del inputs[k]
        self._dirty = True
        return True

    def update_node_inputs(self, node_id: str, updates: dict) -> bool:
        """Merge `updates` into the node's inputs dict."""
        nid = str(node_id)
        if nid not in self._nodes:
            return False
        self._nodes[nid].setdefault("inputs", {}).update(updates)
        self._dirty = True
        return True

    def set_node_title(self, node_id: str, title: str) -> bool:
        nid = str(node_id)
        if nid not in self._nodes:
            return False
        self._nodes[nid].setdefault("_meta", {})["title"] = title
        self._dirty = True
        return True

    # --------------------------------------------------------- link helpers
    def create_link(
        self,
        from_node_id: str,
        from_slot: int,
        to_node_id: str,
        to_input_name: str,
    ) -> bool:
        """Connect output slot `from_slot` of `from_node_id` to the named
        input `to_input_name` of `to_node_id`."""
        fid, tid = str(from_node_id), str(to_node_id)
        if fid not in self._nodes or tid not in self._nodes:
            return False
        self._nodes[tid].setdefault("inputs", {})[to_input_name] = [fid, from_slot]
        self._dirty = True
        return True

    def remove_link(self, to_node_id: str, to_input_name: str) -> bool:
        """Remove the link connected to input `to_input_name` of `to_node_id`."""
        tid = str(to_node_id)
        if tid not in self._nodes:
            return False
        inputs = self._nodes[tid].get("inputs", {})
        if to_input_name not in inputs:
            return False
        v = inputs[to_input_name]
        if not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)):
            return False
        del inputs[to_input_name]
        self._dirty = True
        return True

    # ------------------------------------------------------ file operations
    def save(self, path: str | None = None, name: str | None = None) -> str:
        """Save workflow to file. Returns the saved path."""
        if path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{name or 'workflow'}_{ts}.json"
            path = os.path.join(WORKFLOWS_DIR, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._nodes, f, indent=2)
        self._dirty = False
        return path

    def to_json_string(self) -> str:
        return json.dumps(self._nodes, indent=2)

    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    # -------------------------------------------------------- apply_patch
    def apply_patch(self, patch: dict) -> list[str]:
        """
        Apply a batch of changes in one call.

        patch format:
        {
            "add": [{"class_type": ..., "inputs": ..., "title": ..., "node_id": ...}, ...],
            "remove": ["node_id", ...],
            "update": [{"node_id": ..., "inputs": {...}}, ...],
            "link": [{"from": "id", "from_slot": 0, "to": "id", "to_input": "name"}, ...],
            "unlink": [{"node_id": "id", "input_name": "name"}, ...]
        }

        Returns list of log messages.
        """
        log = []
        for spec in patch.get("add", []):
            nid = self.add_node(
                class_type=spec["class_type"],
                inputs=spec.get("inputs"),
                title=spec.get("title", ""),
                node_id=spec.get("node_id"),
            )
            log.append(f"Added node [{nid}] {spec['class_type']}")

        for nid in patch.get("remove", []):
            if self.remove_node(nid):
                log.append(f"Removed node [{nid}]")
            else:
                log.append(f"WARN: node [{nid}] not found for removal")

        for upd in patch.get("update", []):
            if self.update_node_inputs(upd["node_id"], upd.get("inputs", {})):
                log.append(f"Updated node [{upd['node_id']}]")
            else:
                log.append(f"WARN: node [{upd['node_id']}] not found for update")

        for lnk in patch.get("link", []):
            ok = self.create_link(
                lnk["from"], lnk.get("from_slot", 0),
                lnk["to"], lnk["to_input"]
            )
            log.append(
                f"Linked [{lnk['from']}]:{lnk.get('from_slot',0)} → [{lnk['to']}].{lnk['to_input']}"
                if ok else f"WARN: link failed {lnk}"
            )

        for ul in patch.get("unlink", []):
            ok = self.remove_link(ul["node_id"], ul["input_name"])
            log.append(
                f"Unlinked [{ul['node_id']}].{ul['input_name']}"
                if ok else f"WARN: unlink failed {ul}"
            )

        return log


# ------------------------------------------------------------------ presets
def build_basic_txt2img() -> WorkflowManager:
    """
    Build a minimal text-to-image workflow (SD 1.5 / SDXL compatible).
    Uses only built-in ComfyUI nodes. No custom nodes required.
    """
    wf = WorkflowManager()

    # 1 - Load checkpoint
    wf.add_node("CheckpointLoaderSimple",
                {"ckpt_name": "v1-5-pruned-emaonly.ckpt"},
                "Load Checkpoint", node_id="1")

    # 2 - CLIP text encode (positive)
    wf.add_node("CLIPTextEncode",
                {"text": "beautiful landscape, masterpiece, 8k", "clip": ["1", 1]},
                "Positive Prompt", node_id="2")

    # 3 - CLIP text encode (negative)
    wf.add_node("CLIPTextEncode",
                {"text": "ugly, blurry, watermark", "clip": ["1", 1]},
                "Negative Prompt", node_id="3")

    # 4 - Empty latent image
    wf.add_node("EmptyLatentImage",
                {"width": 512, "height": 512, "batch_size": 1},
                "Empty Latent", node_id="4")

    # 5 - KSampler
    wf.add_node("KSampler",
                {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                 "latent_image": ["4", 0], "seed": 42, "steps": 20,
                 "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
                 "denoise": 1.0},
                "KSampler", node_id="5")

    # 6 - VAE decode
    wf.add_node("VAEDecode",
                {"samples": ["5", 0], "vae": ["1", 2]},
                "VAE Decode", node_id="6")

    # 7 - Save image
    wf.add_node("SaveImage",
                {"images": ["6", 0], "filename_prefix": "ComfyUI"},
                "Save Image", node_id="7")

    return wf


def build_basic_img2img() -> WorkflowManager:
    wf = WorkflowManager()

    wf.add_node("CheckpointLoaderSimple",
                {"ckpt_name": "v1-5-pruned-emaonly.ckpt"},
                "Load Checkpoint", node_id="1")
    wf.add_node("LoadImage",
                {"image": "example.png", "upload": "image"},
                "Load Image", node_id="2")
    wf.add_node("VAEEncode",
                {"pixels": ["2", 0], "vae": ["1", 2]},
                "VAE Encode", node_id="3")
    wf.add_node("CLIPTextEncode",
                {"text": "a beautiful painting", "clip": ["1", 1]},
                "Positive", node_id="4")
    wf.add_node("CLIPTextEncode",
                {"text": "ugly, blurry", "clip": ["1", 1]},
                "Negative", node_id="5")
    wf.add_node("KSampler",
                {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                 "latent_image": ["3", 0], "seed": 42, "steps": 20,
                 "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
                 "denoise": 0.75},
                "KSampler", node_id="6")
    wf.add_node("VAEDecode",
                {"samples": ["6", 0], "vae": ["1", 2]},
                "VAE Decode", node_id="7")
    wf.add_node("SaveImage",
                {"images": ["7", 0], "filename_prefix": "img2img"},
                "Save Image", node_id="8")

    return wf


PRESET_BUILDERS = {
    "txt2img": build_basic_txt2img,
    "img2img": build_basic_img2img,
}
