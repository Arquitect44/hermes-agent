from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-image"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
INTERACTIONS_PATH = "/interactions"

MODELS: Dict[str, Dict[str, str]] = {
    "gemini-3.1-flash-image": {
        "display": "Gemini 3.1 Flash Image",
        "speed": "fast",
        "strengths": "Default, fast image generation with strong text rendering",
        "price": "Google AI Studio billing",
    },
    "gemini-3.1-flash-lite-image": {
        "display": "Gemini 3.1 Flash Lite Image",
        "speed": "fastest",
        "strengths": "Lowest-cost 1K image generation",
        "price": "Google AI Studio billing",
    },
    "gemini-3-pro-image": {
        "display": "Gemini 3 Pro Image",
        "speed": "slower",
        "strengths": "Premium visual quality and complex creative control",
        "price": "Google AI Studio billing",
    },
}

ASPECT_RATIOS: Dict[str, str] = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


def load_image_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except (ImportError, OSError, ValueError) as exc:
        logger.debug("Could not load Hermes image config: %s", exc)
        return {}
    section = cfg.get("image_gen") if isinstance(cfg, dict) else None
    return section if isinstance(section, Mapping) else {}


def env_value(*names: str) -> str:
    try:
        from hermes_cli.config import get_env_value
    except ImportError:
        get_env_value = None
    for name in names:
        value = get_env_value(name) if get_env_value else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def gemini_config() -> Mapping[str, Any]:
    section = load_image_config().get("gemini")
    return section if isinstance(section, Mapping) else {}


def resolve_model() -> str:
    candidate = gemini_config().get("model")
    if isinstance(candidate, str) and candidate in MODELS:
        return candidate
    shared = load_image_config().get("model")
    if isinstance(shared, str) and shared in MODELS:
        return shared
    return DEFAULT_MODEL


def base_url() -> str:
    configured = gemini_config().get("base_url")
    if isinstance(configured, str) and configured.strip():
        return configured.strip().rstrip("/")
    env_base = env_value("GEMINI_BASE_URL")
    return env_base.rstrip("/") if env_base else DEFAULT_BASE_URL


def image_size() -> str:
    configured = gemini_config().get("image_size")
    if isinstance(configured, str) and configured in {"512px", "1K", "2K", "4K"}:
        return configured
    return "1K"


def request_payload(prompt: str, model: str, aspect: str) -> Dict[str, Any]:
    return {
        "model": model,
        "input": [{"type": "text", "text": prompt}],
        "response_format": {
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": ASPECT_RATIOS.get(aspect, "16:9"),
            "image_size": image_size(),
        },
    }


def walk_json(value: Any) -> List[Mapping[str, Any]]:
    found: List[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        found.append(value)
        for child in value.values():
            found.extend(walk_json(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_json(child))
    return found


def extract_image(response_json: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    for item in walk_json(response_json):
        mime_type = item.get("mime_type") or item.get("mimeType")
        data = item.get("data")
        if isinstance(mime_type, str) and mime_type.startswith("image/") and isinstance(data, str):
            extension = mime_type.split("/", 1)[1].split(";", 1)[0] or "png"
            return data, extension

    output_image = response_json.get("output_image")
    if isinstance(output_image, Mapping):
        data = output_image.get("data")
        mime_type = output_image.get("mime_type") or output_image.get("mimeType")
        if isinstance(data, str):
            extension = "png"
            if isinstance(mime_type, str) and mime_type.startswith("image/"):
                extension = mime_type.split("/", 1)[1].split(";", 1)[0] or "png"
            return data, extension

    return None, "png"
