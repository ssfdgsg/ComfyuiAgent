"""
Custom node manager: list, search, install, update ComfyUI custom node packages.

Uses the official ComfyUI Manager registry:
  https://github.com/ltdrdata/ComfyUI-Manager/blob/main/custom-node-list.json

Installation = git clone into custom_nodes/ + pip install -r requirements.txt
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CUSTOM_NODES_PATH, COMFYUI_NODE_LIST_URL


# ------------------------------------------------------------ registry fetch
_cached_registry: list[dict] | None = None


def fetch_node_registry(force: bool = False) -> list[dict]:
    """Fetch (and cache) the ComfyUI Manager custom-node-list.json."""
    global _cached_registry
    if _cached_registry is not None and not force:
        return _cached_registry
    try:
        r = requests.get(COMFYUI_NODE_LIST_URL, timeout=20)
        r.raise_for_status()
        data = r.json()
        _cached_registry = data.get("custom_nodes", data) if isinstance(data, dict) else data
    except Exception as e:
        _cached_registry = []
        return [{"error": str(e)}]
    return _cached_registry


# --------------------------------------------------------- local inspection
def list_installed_packages() -> list[dict]:
    """
    Return list of installed custom node packages.
    Each entry: {name, path, has_requirements, git_remote}.
    """
    if not os.path.isdir(CUSTOM_NODES_PATH):
        return []
    packages = []
    for entry in sorted(os.listdir(CUSTOM_NODES_PATH)):
        full = os.path.join(CUSTOM_NODES_PATH, entry)
        if not os.path.isdir(full) or entry.startswith("."):
            continue
        git_remote = ""
        git_config = os.path.join(full, ".git", "config")
        if os.path.exists(git_config):
            with open(git_config) as f:
                for line in f:
                    if "url = " in line:
                        git_remote = line.split("url = ", 1)[1].strip()
                        break
        packages.append({
            "name": entry,
            "path": full,
            "has_requirements": os.path.exists(os.path.join(full, "requirements.txt")),
            "git_remote": git_remote,
        })
    return packages


def get_installed_summary() -> str:
    """Short text summary of installed custom node packages."""
    pkgs = list_installed_packages()
    if not pkgs:
        return "No custom nodes installed."
    names = [p["name"] for p in pkgs]
    return f"Installed custom node packages ({len(pkgs)}): {', '.join(names)}"


# ------------------------------------------------------------------- search
def search_nodes(query: str, limit: int = 10) -> list[dict]:
    """
    Search the ComfyUI Manager registry.
    Returns list of {title, description, install_type, reference (git URL), pip}.
    """
    registry = fetch_node_registry()
    if registry and isinstance(registry[0], dict) and "error" in registry[0]:
        return registry

    q = query.lower()
    results = []
    for node in registry:
        title = node.get("title", "").lower()
        desc = node.get("description", "").lower()
        if q in title or q in desc:
            results.append({
                "title": node.get("title"),
                "description": node.get("description", "")[:200],
                "install_type": node.get("install_type", "git-clone"),
                "reference": node.get("reference", ""),
                "pip": node.get("pip", []),
                "files": node.get("files", []),
            })
        if len(results) >= limit:
            break
    return results


def get_node_by_title(title: str) -> dict | None:
    """Find a node entry in the registry by exact title (case-insensitive)."""
    registry = fetch_node_registry()
    for node in registry:
        if node.get("title", "").lower() == title.lower():
            return node
    return None


# ----------------------------------------------------------------- install
def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def install_custom_node(git_url: str, pip_packages: list[str] | None = None) -> dict:
    """
    Install a custom node package by git URL.

    Steps:
      1. git clone into custom_nodes/<repo_name>
      2. pip install -r requirements.txt  (if present)
      3. pip install <extra pip packages> (if any)

    Returns {success, name, path, log, error}.
    """
    os.makedirs(CUSTOM_NODES_PATH, exist_ok=True)

    repo_name = git_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    dest = os.path.join(CUSTOM_NODES_PATH, repo_name)
    log_lines: list[str] = []

    if os.path.exists(dest):
        return {
            "success": False,
            "name": repo_name,
            "path": dest,
            "log": "",
            "error": f"Already installed at {dest}. Use update_custom_node to update.",
        }

    # Clone
    rc, out, err = _run(["git", "clone", "--depth", "1", git_url, dest])
    log_lines.append(f"git clone: rc={rc}\n{out}\n{err}")
    if rc != 0:
        return {"success": False, "name": repo_name, "path": "", "log": "\n".join(log_lines), "error": err}

    # pip install requirements.txt
    req_file = os.path.join(dest, "requirements.txt")
    if os.path.exists(req_file):
        rc2, out2, err2 = _run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"]
        )
        log_lines.append(f"pip install requirements: rc={rc2}\n{err2}")

    # Extra pip packages from registry
    if pip_packages:
        rc3, out3, err3 = _run(
            [sys.executable, "-m", "pip", "install"] + pip_packages + ["--quiet"]
        )
        log_lines.append(f"pip install extras: rc={rc3}\n{err3}")

    return {
        "success": True,
        "name": repo_name,
        "path": dest,
        "log": "\n".join(log_lines),
        "error": "",
    }


def install_node_by_title(title: str) -> dict:
    """Look up a node in the registry by title and install it."""
    node = get_node_by_title(title)
    if not node:
        return {"success": False, "error": f"Node '{title}' not found in registry."}
    ref = node.get("reference", "")
    if not ref:
        return {"success": False, "error": "No git reference URL in registry entry."}
    return install_custom_node(ref, pip_packages=node.get("pip", []))


def update_custom_node(name: str) -> dict:
    """git pull on an installed custom node package."""
    dest = os.path.join(CUSTOM_NODES_PATH, name)
    if not os.path.isdir(dest):
        return {"success": False, "error": f"Package '{name}' not found at {dest}"}

    rc, out, err = _run(["git", "pull"], cwd=dest)
    log = f"git pull: rc={rc}\n{out}\n{err}"

    if rc == 0:
        req_file = os.path.join(dest, "requirements.txt")
        if os.path.exists(req_file):
            _run([sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"])
    return {"success": rc == 0, "log": log, "error": err if rc != 0 else ""}


def uninstall_custom_node(name: str) -> dict:
    """Remove a custom node package directory."""
    import shutil
    dest = os.path.join(CUSTOM_NODES_PATH, name)
    if not os.path.isdir(dest):
        return {"success": False, "error": f"Package '{name}' not found."}
    shutil.rmtree(dest)
    return {"success": True, "message": f"Removed {dest}"}
