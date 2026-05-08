"""
Model manager: local scanning, HuggingFace search, CivitAI search, downloading.

Download always streams with tqdm so large models (10+ GB) work without
exhausting RAM.
"""
from __future__ import annotations

import os
import re
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    MODEL_DIRS, MODELS_BASE,
    HUGGINGFACE_TOKEN, CIVITAI_API_KEY, MODELSCOPE_TOKEN,
    COMFYUI_MODEL_LIST_URL,
    DOWNLOAD_CHUNK_SIZE, DOWNLOAD_TIMEOUT,
)

# ----------------------------------------------------------------- scanning
def scan_local_models() -> dict[str, list[dict]]:
    """
    Walk the models directory and return a dict keyed by model category.
    Each entry is a list of {name, size_mb, path}.
    """
    result: dict[str, list[dict]] = {}
    for category, dir_path in MODEL_DIRS.items():
        if not os.path.isdir(dir_path):
            continue
        files = []
        for root, _dirs, fnames in os.walk(dir_path):
            for fname in fnames:
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in {".safetensors", ".ckpt", ".pt", ".pth", ".bin",
                                ".onnx", ".gguf", ".pkl"}:
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, dir_path)
                size_mb = round(os.path.getsize(full) / 1024 / 1024, 1)
                files.append({"name": rel, "size_mb": size_mb, "path": full})
        if files:
            result[category] = sorted(files, key=lambda x: x["name"])
    return result


def get_model_summary() -> str:
    """Short text summary of installed models — safe to put in LLM context."""
    local = scan_local_models()
    if not local:
        return "No models installed yet."
    lines = ["Installed models:"]
    for cat, files in local.items():
        total_gb = sum(f["size_mb"] for f in files) / 1024
        names = [f["name"] for f in files]
        lines.append(f"  {cat} ({len(files)} files, {total_gb:.1f} GB): {', '.join(names)}")
    return "\n".join(lines)


# -------------------------------------------------------- HuggingFace search
HF_API = "https://huggingface.co/api"


def search_huggingface(
    query: str,
    model_type: str = "",
    limit: int = 10,
) -> list[dict]:
    """
    Search HuggingFace Hub for models.

    Returns list of {repo_id, pipeline_tag, downloads, likes, url, description}.
    """
    params: dict = {"search": query, "limit": limit, "sort": "downloads"}
    if model_type:
        params["pipeline_tag"] = model_type
    headers = {}
    if HUGGINGFACE_TOKEN:
        headers["Authorization"] = f"Bearer {HUGGINGFACE_TOKEN}"

    try:
        r = requests.get(f"{HF_API}/models", params=params, headers=headers, timeout=20)
        r.raise_for_status()
        results = r.json()
    except Exception as e:
        return [{"error": str(e)}]

    out = []
    for m in results[:limit]:
        out.append({
            "repo_id": m.get("modelId", ""),
            "pipeline_tag": m.get("pipeline_tag", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "url": f"https://huggingface.co/{m.get('modelId', '')}",
            "description": (m.get("cardData") or {}).get("language", ""),
        })
    return out


def get_hf_model_files(repo_id: str) -> list[dict]:
    """List downloadable files for a HuggingFace repo."""
    headers = {}
    if HUGGINGFACE_TOKEN:
        headers["Authorization"] = f"Bearer {HUGGINGFACE_TOKEN}"
    try:
        r = requests.get(f"{HF_API}/models/{repo_id}", headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [{"error": str(e)}]

    siblings = data.get("siblings", [])
    files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        ext = os.path.splitext(fname)[1].lower()
        if ext in {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}:
            files.append({
                "filename": fname,
                "size_bytes": s.get("size", 0),
                "download_url": f"https://huggingface.co/{repo_id}/resolve/main/{fname}",
            })
    return files


# ----------------------------------------------------------- CivitAI search
CIVITAI_API = "https://civitai.com/api/v1"

CIVITAI_MODEL_TYPES = {
    "checkpoint": "Checkpoint",
    "lora": "LORA",
    "lycoris": "LyCORIS",
    "textual_inversion": "TextualInversion",
    "hypernetwork": "Hypernetwork",
    "controlnet": "Controlnet",
    "vae": "VAE",
    "upscaler": "Upscaler",
}


def search_civitai(
    query: str,
    model_type: str = "",
    limit: int = 10,
    nsfw: bool = False,
) -> list[dict]:
    """
    Search CivitAI for models.
    Returns list of {id, name, type, downloads, rating, url, versions}.
    """
    params: dict = {"query": query, "limit": limit, "nsfw": str(nsfw).lower()}
    if model_type:
        params["types"] = CIVITAI_MODEL_TYPES.get(model_type.lower(), model_type)
    headers = {}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"

    try:
        r = requests.get(f"{CIVITAI_API}/models", params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [{"error": str(e)}]

    out = []
    for m in data.get("items", [])[:limit]:
        versions = []
        for v in m.get("modelVersions", [])[:3]:
            files = [
                {
                    "filename": f.get("name"),
                    "size_kb": f.get("sizeKB"),
                    "download_url": f.get("downloadUrl"),
                }
                for f in v.get("files", [])
                if f.get("primary")
            ]
            versions.append({"name": v.get("name"), "files": files})
        out.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "type": m.get("type"),
            "downloads": m.get("stats", {}).get("downloadCount", 0),
            "rating": m.get("stats", {}).get("rating", 0),
            "url": f"https://civitai.com/models/{m.get('id')}",
            "versions": versions,
        })
    return out


# ---------------------------------------------------------- ComfyUI Manager model list
def get_comfyui_model_list() -> list[dict]:
    """Fetch the official ComfyUI Manager model list."""
    try:
        r = requests.get(COMFYUI_MODEL_LIST_URL, timeout=20)
        r.raise_for_status()
        return r.json().get("models", [])
    except Exception as e:
        return [{"error": str(e)}]


# --------------------------------------------------------------- downloading
class DownloadProgress:
    """Thread-safe download progress tracker."""
    def __init__(self):
        self._lock = threading.Lock()
        self.active: dict[str, dict] = {}  # filename -> {total, downloaded, done, error}

    def start(self, fname: str, total: int):
        with self._lock:
            self.active[fname] = {"total": total, "downloaded": 0, "done": False, "error": ""}

    def update(self, fname: str, chunk: int):
        with self._lock:
            if fname in self.active:
                self.active[fname]["downloaded"] += chunk

    def finish(self, fname: str, error: str = ""):
        with self._lock:
            if fname in self.active:
                self.active[fname]["done"] = True
                self.active[fname]["error"] = error

    def status(self, fname: str) -> dict:
        with self._lock:
            return dict(self.active.get(fname, {}))

    def all_status(self) -> dict:
        with self._lock:
            return dict(self.active)


_progress = DownloadProgress()


def get_download_progress() -> dict:
    return _progress.all_status()


def download_model(
    url: str,
    category: str,
    filename: str | None = None,
    headers_extra: dict | None = None,
) -> dict:
    """
    Download a model file to the correct ComfyUI models sub-directory.

    - Streams the download; works for files of any size.
    - Shows tqdm progress bar.
    - Returns {success, path, size_mb, error}.
    """
    if category not in MODEL_DIRS:
        return {"success": False, "error": f"Unknown category '{category}'. Valid: {list(MODEL_DIRS.keys())}"}

    dest_dir = MODEL_DIRS[category]
    os.makedirs(dest_dir, exist_ok=True)

    # Determine filename from URL if not provided
    if not filename:
        filename = url.split("?")[0].split("/")[-1]
    if not filename:
        filename = f"model_{int(time.time())}.bin"

    dest_path = os.path.join(dest_dir, filename)

    headers: dict[str, str] = {}
    if HUGGINGFACE_TOKEN and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {HUGGINGFACE_TOKEN}"
    if CIVITAI_API_KEY and "civitai.com" in url:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    if headers_extra:
        headers.update(headers_extra)

    _progress.start(filename, 0)
    try:
        with requests.get(url, headers=headers, stream=True,
                          timeout=DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            _progress.start(filename, total)

            with open(dest_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=filename, ncols=80
            ) as bar:
                for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
                        _progress.update(filename, len(chunk))

        size_mb = round(os.path.getsize(dest_path) / 1024 / 1024, 1)
        _progress.finish(filename)
        return {"success": True, "path": dest_path, "size_mb": size_mb, "error": ""}
    except Exception as e:
        _progress.finish(filename, str(e))
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return {"success": False, "path": "", "size_mb": 0, "error": str(e)}


def download_model_background(
    url: str, category: str, filename: str | None = None
) -> str:
    """Start download in a background thread. Returns filename tracking key."""
    fname = filename or url.split("?")[0].split("/")[-1]
    t = threading.Thread(
        target=download_model,
        args=(url, category, fname),
        daemon=True,
    )
    t.start()
    return fname


# ---------------------------------------------------------- ModelScope (魔搭)
MODELSCOPE_API = "https://modelscope.cn/api/v1"


def search_modelscope(query: str, limit: int = 10) -> list[dict]:
    """
    Search ModelScope (魔搭) for models.
    Returns list of {model_id, name, downloads, url, task}.
    """
    headers = {}
    if MODELSCOPE_TOKEN:
        headers["Authorization"] = f"Token {MODELSCOPE_TOKEN}"
    try:
        r = requests.get(
            f"{MODELSCOPE_API}/models",
            params={"Name": query, "PageSize": limit, "SortBy": "Downloads"},
            headers=headers,
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [{"error": str(e)}]

    results = []
    for m in data.get("Data", {}).get("Models", [])[:limit]:
        results.append({
            "model_id": m.get("Path", ""),
            "name": m.get("Name", ""),
            "downloads": m.get("Downloads", 0),
            "url": f"https://modelscope.cn/models/{m.get('Path', '')}",
            "task": m.get("Tasks", [{}])[0].get("Name", "") if m.get("Tasks") else "",
        })
    return results


def download_from_modelscope(
    model_id: str,
    category: str,
    file_pattern: str = "*.safetensors",
) -> dict:
    """
    Download a model from ModelScope to the correct ComfyUI models directory.

    Uses the modelscope SDK snapshot_download under the hood.
    model_id example: 'AI-ModelScope/stable-diffusion-xl-base-1.0'
    """
    if category not in MODEL_DIRS:
        return {"success": False, "error": f"Unknown category: {category}"}

    dest_dir = MODEL_DIRS[category]
    os.makedirs(dest_dir, exist_ok=True)

    try:
        from modelscope import snapshot_download
        from modelscope.hub.api import HubApi
    except ImportError:
        return {"success": False, "error": "modelscope package not installed. Run: pip install modelscope"}

    try:
        if MODELSCOPE_TOKEN:
            api = HubApi()
            api.login(MODELSCOPE_TOKEN)

        local_dir = snapshot_download(
            model_id=model_id,
            local_dir=dest_dir,
            ignore_patterns=["*.bin", "*.ot", "*.msgpack", "flax_*", "tf_*"]
            if file_pattern == "*.safetensors"
            else [],
        )
        # Compute total size
        total_mb = sum(
            os.path.getsize(os.path.join(r, f)) / 1024 / 1024
            for r, _, files in os.walk(local_dir)
            for f in files
        )
        return {
            "success": True,
            "path": local_dir,
            "size_mb": round(total_mb, 1),
            "error": "",
        }
    except Exception as e:
        return {"success": False, "path": "", "size_mb": 0, "error": str(e)}
