"""LoadImageFromURL — fetch a remote image and return ComfyUI IMAGE + MASK tensors.

Also previews the fetched image inside the node (like LoadImage / PreviewImage).
"""
import io
import os
import random
import time
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import folder_paths  # provided by ComfyUI at runtime
except ImportError:
    folder_paths = None


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "ComfyUI-DioBrando-Nodes/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _pil_to_tensors(pil: Image.Image):
    pil = ImageOps.exif_transpose(pil)
    has_alpha = pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info)

    if has_alpha:
        rgba = pil.convert("RGBA")
        rgb = np.array(rgba.convert("RGB"), dtype=np.float32) / 255.0
        alpha = np.array(rgba.split()[-1], dtype=np.float32) / 255.0
        mask = 1.0 - alpha
    else:
        rgb = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
        mask = np.zeros(rgb.shape[:2], dtype=np.float32)

    image_t = torch.from_numpy(rgb).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    return image_t, mask_t


def _save_preview(pil: Image.Image) -> dict:
    """Save image to ComfyUI temp dir and return the UI dict for inline preview."""
    if folder_paths is None:
        return {}
    pil_rgb = pil.convert("RGB") if pil.mode != "RGB" else pil
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"db_loadurl_{int(time.time() * 1000)}_{random.randint(0, 9999)}.png"
    pil_rgb.save(os.path.join(temp_dir, filename), format="PNG")
    return {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}


class LoadImageFromURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/image.png", "multiline": False}),
            },
            "optional": {
                "timeout": ("INT", {"default": 60, "min": 5, "max": 600, "step": 5}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, url, timeout=60):
        return url

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "url")
    FUNCTION = "load"
    CATEGORY = "DioBrando/IO"
    OUTPUT_NODE = True

    def load(self, url, timeout=60):
        if not url or not url.strip():
            raise RuntimeError("LoadImageFromURL: empty URL")
        data = _fetch(url.strip(), timeout=timeout)
        pil = Image.open(io.BytesIO(data))
        image_t, mask_t = _pil_to_tensors(pil)
        return {
            "ui": _save_preview(pil),
            "result": (image_t, mask_t, url.strip()),
        }
