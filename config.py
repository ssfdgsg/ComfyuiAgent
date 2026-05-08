"""
Central configuration. All secrets come from environment variables.
"""
import os

# ComfyUI service
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
COMFYUI_PATH = os.getenv("COMFYUI_PATH", "/root/ComfyUI")
MODELS_BASE = os.path.join(COMFYUI_PATH, "models")
CUSTOM_NODES_PATH = os.path.join(COMFYUI_PATH, "custom_nodes")
INPUT_PATH = os.path.join(COMFYUI_PATH, "input")
OUTPUT_PATH = os.path.join(COMFYUI_PATH, "output")

# Agent workspace
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(AGENT_DIR, "state")
WORKFLOWS_DIR = os.path.join(AGENT_DIR, "workflows")

# State documents
RESOURCES_DOC = os.path.join(STATE_DIR, "resources.md")
WORKFLOW_STATE_DOC = os.path.join(STATE_DIR, "workflow_state.md")
HISTORY_LOG = os.path.join(STATE_DIR, "history.jsonl")

# ── LLM providers ──────────────────────────────────────────────────────────
# Primary: Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# Fallback: any OpenAI-compatible endpoint
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Which provider to use by default ("gemini" | "openai")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "gemini")

# Agent loop limits
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "40"))

# ── Model hub tokens ────────────────────────────────────────────────────────
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "")
MODELSCOPE_TOKEN = os.getenv("MODELSCOPE_TOKEN", "")  # 魔搭 API token

# ── Web UI ──────────────────────────────────────────────────────────────────
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))

# ── Model directory layout ──────────────────────────────────────────────────
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
    "hypernetworks": os.path.join(MODELS_BASE, "hypernetworks"),
    "photomaker": os.path.join(MODELS_BASE, "photomaker"),
    "instantid": os.path.join(MODELS_BASE, "instantid"),
}

# ── External registries ─────────────────────────────────────────────────────
COMFYUI_NODE_LIST_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"
)
COMFYUI_MODEL_LIST_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"
)

# ── Download settings ───────────────────────────────────────────────────────
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB
DOWNLOAD_TIMEOUT = 30
