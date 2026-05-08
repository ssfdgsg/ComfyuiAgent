"""
Central configuration for ComfyUI Agent.
All paths are relative to the yanwk/comfyui-boot container layout.
"""
import os

# ComfyUI service
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
COMFYUI_PATH = os.getenv("COMFYUI_PATH", "/root/ComfyUI")
MODELS_BASE = os.path.join(COMFYUI_PATH, "models")
CUSTOM_NODES_PATH = os.path.join(COMFYUI_PATH, "custom_nodes")
INPUT_PATH = os.path.join(COMFYUI_PATH, "input")
OUTPUT_PATH = os.path.join(COMFYUI_PATH, "output")

# Agent state & workspace (written alongside the agent code)
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(AGENT_DIR, "state")
WORKFLOWS_DIR = os.path.join(AGENT_DIR, "workflows")

# State documents
RESOURCES_DOC = os.path.join(STATE_DIR, "resources.md")
WORKFLOW_STATE_DOC = os.path.join(STATE_DIR, "workflow_state.md")
HISTORY_LOG = os.path.join(STATE_DIR, "history.jsonl")

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "40"))

# Model directories (standard ComfyUI layout)
MODEL_DIRS = {
    "checkpoints": os.path.join(MODELS_BASE, "checkpoints"),
    "vae": os.path.join(MODELS_BASE, "vae"),
    "loras": os.path.join(MODELS_BASE, "loras"),
    "controlnet": os.path.join(MODELS_BASE, "controlnet"),
    "embeddings": os.path.join(MODELS_BASE, "embeddings"),
    "upscale_models": os.path.join(MODELS_BASE, "upscale_models"),
    "clip": os.path.join(MODELS_BASE, "clip"),
    "unet": os.path.join(MODELS_BASE, "unet"),
    "clip_vision": os.path.join(MODELS_BASE, "clip_vision"),
    "ipadapter": os.path.join(MODELS_BASE, "ipadapter"),
    "style_models": os.path.join(MODELS_BASE, "style_models"),
    "diffusers": os.path.join(MODELS_BASE, "diffusers"),
    "gligen": os.path.join(MODELS_BASE, "gligen"),
    "hypernetworks": os.path.join(MODELS_BASE, "hypernetworks"),
    "photomaker": os.path.join(MODELS_BASE, "photomaker"),
    "instantid": os.path.join(MODELS_BASE, "instantid"),
    "onnx": os.path.join(MODELS_BASE, "onnx"),
}

# External API keys (optional, improves search quality)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "")

# ComfyUI Manager custom node list (official registry)
COMFYUI_NODE_LIST_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"
)
COMFYUI_MODEL_LIST_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"
)

# Download settings
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB
DOWNLOAD_TIMEOUT = 30  # seconds per chunk before giving up
