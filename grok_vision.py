"""Grok vision analyze nodes (image tensor + URL variants)."""
import os
import io
import json
import base64
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import torch
from PIL import Image


XAI_ENDPOINT = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4.3-latest"

VISION_MODELS = [
    "grok-4.3-latest",
    "grok-4.3",
    "grok-4.20-multi-agent-0309",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-4-vision",
    "grok-3-vision-beta",
    "grok-2-vision-1212",
    "grok-vision-beta",
]


def _tensor_to_data_url(image_tensor: torch.Tensor, fmt: str = "JPEG", quality: int = 90) -> str:
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    img = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil.save(buf, format=fmt, quality=quality if fmt.upper() == "JPEG" else None)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _resolve_api_key(override: str) -> str:
    if override and override.strip():
        return override.strip()
    for env_name in ("XAI_API", "XAI_API_KEY", "GROK_API_KEY"):
        v = os.environ.get(env_name)
        if v:
            return v.strip()
    raise RuntimeError(
        "No xAI API key found. Set XAI_API in your shell environment, or paste the key into the api_key input."
    )


def _call_grok(api_key: str, model: str, prompt: str, image_url: str,
               max_tokens: int, temperature: float, detail: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    req = Request(
        XAI_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode(errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"xAI API error {e.code}: {body}") from e


def _format_usage(usage: dict) -> str:
    return (
        f"prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')} "
        f"total={usage.get('total_tokens')}"
    )


class GrokVisionAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                }),
                "model": (VISION_MODELS, {"default": DEFAULT_MODEL}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 100000, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "detail": (["high", "low", "auto"], {"default": "high"}),
                "image_format": (["JPEG", "PNG"], {"default": "JPEG"}),
                "jpeg_quality": ("INT", {"default": 90, "min": 60, "max": 100, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "leave empty to use XAI_API env var"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "usage")
    FUNCTION = "analyze"
    CATEGORY = "DioBrando/Grok"

    def analyze(self, image, prompt, model, max_tokens, temperature, detail,
                image_format, jpeg_quality, api_key=""):
        key = _resolve_api_key(api_key)
        data_url = _tensor_to_data_url(image, fmt=image_format, quality=jpeg_quality)
        result = _call_grok(key, model, prompt, data_url, max_tokens, temperature, detail)
        msg = result["choices"][0]["message"]["content"]
        return (msg, _format_usage(result.get("usage", {})))


class GrokVisionAnalyzeURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"default": "https://example.com/image.jpg", "multiline": False, "forceInput": True}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "model": (VISION_MODELS, {"default": DEFAULT_MODEL}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 100000, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "detail": (["high", "low", "auto"], {"default": "high"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "leave empty to use XAI_API env var"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "usage")
    FUNCTION = "analyze"
    CATEGORY = "DioBrando/Grok"

    def analyze(self, image_url, prompt, model, max_tokens, temperature, detail, api_key=""):
        key = _resolve_api_key(api_key)
        result = _call_grok(key, model, prompt, image_url, max_tokens, temperature, detail)
        msg = result["choices"][0]["message"]["content"]
        return (msg, _format_usage(result.get("usage", {})))
