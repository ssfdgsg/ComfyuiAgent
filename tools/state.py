"""
State documentation manager.

Writes three persistent Markdown / JSONL files to the state/ directory:

  resources.md      — installed models and custom nodes (auto-refreshed)
  workflow_state.md — current in-memory workflow description
  history.jsonl     — append-only operation log

These files serve as the agent's "external memory" so long workflows and
large installed model sets don't need to be re-scanned on every turn.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import STATE_DIR, RESOURCES_DOC, WORKFLOW_STATE_DOC, HISTORY_LOG


def _ensure_dirs():
    os.makedirs(STATE_DIR, exist_ok=True)


# -------------------------------------------------------------- resources.md
def refresh_resources_doc() -> str:
    """
    Scan installed models and custom nodes, write resources.md, return its content.
    This is the canonical "what do we have" document.
    """
    _ensure_dirs()

    # Lazy imports to avoid circular dependencies at module load time
    from tools.models import scan_local_models
    from tools.nodes import list_installed_packages

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# ComfyUI Resources  _(last updated: {ts})_\n",
        "## Installed Models\n",
    ]

    models = scan_local_models()
    if not models:
        lines.append("_No models installed._\n")
    else:
        for category, files in models.items():
            total_gb = sum(f["size_mb"] for f in files) / 1024
            lines.append(f"### {category}  ({len(files)} files, {total_gb:.2f} GB)\n")
            for f in files:
                lines.append(f"- `{f['name']}` — {f['size_mb']} MB\n")
        lines.append("\n")

    lines.append("## Installed Custom Nodes\n")
    pkgs = list_installed_packages()
    if not pkgs:
        lines.append("_No custom nodes installed._\n")
    else:
        for p in pkgs:
            remote = f"  ← {p['git_remote']}" if p["git_remote"] else ""
            lines.append(f"- **{p['name']}**{remote}\n")

    content = "".join(lines)
    with open(RESOURCES_DOC, "w") as f:
        f.write(content)
    return content


def read_resources_doc() -> str:
    """Read cached resources.md; refresh if it doesn't exist yet."""
    if not os.path.exists(RESOURCES_DOC):
        return refresh_resources_doc()
    with open(RESOURCES_DOC) as f:
        return f.read()


# --------------------------------------------------------- workflow_state.md
def update_workflow_state(
    summary: str,
    note: str = "",
    workflow_file: str = "",
) -> None:
    """Write the current workflow state to workflow_state.md."""
    _ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Workflow State  _(updated: {ts})_\n\n",
    ]
    if workflow_file:
        lines.append(f"**File:** `{workflow_file}`\n\n")
    if note:
        lines.append(f"**Note:** {note}\n\n")
    lines.append("## Current Workflow\n\n")
    lines.append("```\n")
    lines.append(summary + "\n")
    lines.append("```\n")
    with open(WORKFLOW_STATE_DOC, "w") as f:
        f.writelines(lines)


def read_workflow_state() -> str:
    if not os.path.exists(WORKFLOW_STATE_DOC):
        return "No workflow state recorded yet."
    with open(WORKFLOW_STATE_DOC) as f:
        return f.read()


# --------------------------------------------------------------- history.jsonl
def log_operation(op_type: str, details: dict) -> None:
    """Append an operation record to history.jsonl."""
    _ensure_dirs()
    record = {
        "ts": time.time(),
        "iso": datetime.now().isoformat(),
        "type": op_type,
        **details,
    }
    with open(HISTORY_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_recent_history(n: int = 20) -> list[dict]:
    if not os.path.exists(HISTORY_LOG):
        return []
    records = []
    with open(HISTORY_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[-n:]


def history_summary(n: int = 15) -> str:
    records = read_recent_history(n)
    if not records:
        return "No operations recorded yet."
    lines = [f"Recent {len(records)} operations:"]
    for r in records:
        iso = r.get("iso", "")[:19]
        op = r.get("type", "?")
        detail = {k: v for k, v in r.items() if k not in ("ts", "iso", "type")}
        dstr = json.dumps(detail, ensure_ascii=False)
        if len(dstr) > 120:
            dstr = dstr[:117] + "..."
        lines.append(f"  [{iso}] {op}: {dstr}")
    return "\n".join(lines)


# ---------------------------------------------------------------- full context
def get_agent_context() -> str:
    """
    Assemble a concise context block for the system prompt.
    Keeps the LLM informed of current state without blowing the context window.
    """
    sections = [
        "=== AGENT STATE CONTEXT ===\n",
        read_resources_doc(),
        "\n",
        read_workflow_state(),
        "\n",
        history_summary(),
        "\n=== END CONTEXT ===",
    ]
    return "\n".join(sections)
