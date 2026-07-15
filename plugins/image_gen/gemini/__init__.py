from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)
from plugins.image_gen.gemini.client import (
    DEFAULT_MODEL,
    INTERACTIONS_PATH,
    MODELS,
    base_url,
    env_value,
    extract_image,
    image_size,
    request_payload,
    resolve_model,
)

def _provider_error(
    *,
    error: str,
    error_type: str,
    model: str,
    prompt: str = "",
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> Dict[str, Any]:
    return error_response(
        error=error,
        error_type=error_type,
        provider="gemini",
        model=model,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
    )


class GeminiImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Gemini"

    def is_available(self) -> bool:
        return bool(env_value("GEMINI_API_KEY", "GOOGLE_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": model_id, **meta} for model_id, meta in MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Gemini",
            "badge": "paid",
            "tag": "Google Gemini image generation through the Interactions API",
            "env_vars": [
                {
                    "key": "GEMINI_API_KEY",
                    "prompt": "Gemini API key",
                    "url": "https://aistudio.google.com/app/apikey",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model = resolve_model()

        if not prompt:
            return _provider_error(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                model=model,
                aspect_ratio=aspect,
            )

        api_key = env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not api_key:
            return _provider_error(
                error="GEMINI_API_KEY or GOOGLE_API_KEY is not set in the Hermes environment",
                error_type="auth_required",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        endpoint = f"{base_url()}{INTERACTIONS_PATH}"
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                json=request_payload(prompt, model, aspect),
                timeout=90,
            )
        except requests.RequestException as exc:
            return _provider_error(
                error=f"Gemini image request failed: {exc}",
                error_type="network_error",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if response.status_code >= 400:
            detail = response.text[:300]
            try:
                parsed = response.json()
                if isinstance(parsed, Mapping):
                    error = parsed.get("error")
                    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
                        detail = error["message"][:300]
            except requests.JSONDecodeError:
                pass
            return _provider_error(
                error=f"Gemini image API error (HTTP {response.status_code}): {detail}",
                error_type="api_error",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            response_json = response.json()
        except requests.JSONDecodeError as exc:
            return _provider_error(
                error=f"Gemini image API returned non-JSON response: {exc}",
                error_type="parse_error",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not isinstance(response_json, Mapping):
            return _provider_error(
                error="Gemini image API returned an unexpected response shape",
                error_type="parse_error",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_b64, extension = extract_image(response_json)
        if not image_b64:
            return _provider_error(
                error="Gemini image API returned no image data",
                error_type="empty_response",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(image_b64, prefix=f"gemini_{model}", extension=extension)
        except (OSError, ValueError) as exc:
            return _provider_error(
                error=f"Could not save Gemini image to cache: {exc}",
                error_type="io_error",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="gemini",
            extra={"image_size": image_size()},
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(GeminiImageGenProvider())
