"""
ComfyUI REST API client.

ComfyUI exposes a WebSocket + HTTP API at port 8188. This module wraps
all HTTP endpoints the agent needs to interact with.
"""
import json
import uuid
import time
import requests
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COMFYUI_URL


class ComfyUIClient:
    def __init__(self, base_url: str = COMFYUI_URL):
        self.base = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _get(self, path: str, **kwargs) -> Any:
        r = self._session.get(f"{self.base}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict, **kwargs) -> Any:
        r = self._session.post(
            f"{self.base}{path}", json=payload, timeout=30, **kwargs
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ info
    def is_alive(self) -> bool:
        try:
            self._get("/system_stats")
            return True
        except Exception:
            return False

    def wait_until_ready(self, timeout: int = 60) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_alive():
                return True
            time.sleep(2)
        return False

    def get_system_stats(self) -> dict:
        return self._get("/system_stats")

    def get_object_info(self) -> dict:
        """Return full node-type registry (all available node classes)."""
        return self._get("/object_info")

    def get_object_info_node(self, node_type: str) -> dict:
        """Return definition for a single node type."""
        return self._get(f"/object_info/{node_type}")

    def get_embeddings(self) -> list[str]:
        return self._get("/embeddings")

    def get_extensions(self) -> list[str]:
        return self._get("/extensions")

    # ----------------------------------------------------------------- queue
    def queue_prompt(self, workflow: dict) -> dict:
        """Queue a workflow (API-format) for execution.
        Returns {"prompt_id": "...", "number": N, "node_errors": {...}}.
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        return self._post("/prompt", payload)

    def get_queue(self) -> dict:
        return self._get("/queue")

    def clear_queue(self) -> dict:
        return self._post("/queue", {"clear": True})

    def interrupt(self) -> None:
        self._post("/interrupt", {})

    # --------------------------------------------------------------- history
    def get_history(self, max_items: int = 20) -> dict:
        data = self._get("/history")
        # Return most recent N items
        items = list(data.items())[-max_items:]
        return dict(items)

    def get_prompt_result(self, prompt_id: str) -> dict | None:
        history = self._get("/history")
        return history.get(prompt_id)

    # ----------------------------------------------------------------- files
    def upload_image(self, file_path: str, subfolder: str = "") -> dict:
        with open(file_path, "rb") as f:
            files = {"image": (os.path.basename(file_path), f, "image/png")}
            data = {"overwrite": "true"}
            if subfolder:
                data["subfolder"] = subfolder
            r = self._session.post(
                f"{self.base}/upload/image", files=files, data=data, timeout=60
            )
            r.raise_for_status()
            return r.json()

    def get_image(self, filename: str, subfolder: str = "", image_type: str = "output"):
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        r = self._session.get(f"{self.base}/view", params=params, timeout=60)
        r.raise_for_status()
        return r.content

    # ----------------------------------------------------------- convenience
    def list_available_node_types(self) -> list[str]:
        """Sorted list of all node class names registered in ComfyUI."""
        info = self.get_object_info()
        return sorted(info.keys())

    def poll_until_done(self, prompt_id: str, timeout: int = 600) -> dict | None:
        """Block until the given prompt finishes or timeout is reached."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_prompt_result(prompt_id)
            if result and result.get("outputs"):
                return result
            # Check if still in queue
            queue = self.get_queue()
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            if not any(item[1] == prompt_id for item in running + pending):
                # Not in queue; might have errored or finished
                result = self.get_prompt_result(prompt_id)
                return result
            time.sleep(2)
        return None


# Module-level singleton for convenience
_client: ComfyUIClient | None = None


def get_client() -> ComfyUIClient:
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client
