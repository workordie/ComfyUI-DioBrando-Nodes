"""ComfyUI-DioBrando-Nodes: Grok vision + URL image loader + VAE grid fix.

Author: @diobrando0 (HuggingFace) / @workordie (Civitai)
Repo:   https://github.com/DanielBartolic/ComfyUI-DioBrando-Nodes
"""
from .grok_vision import GrokVisionAnalyze, GrokVisionAnalyzeURL
from .load_image_url import LoadImageFromURL
from .nyquist_notch import NyquistNotch

NODE_CLASS_MAPPINGS = {
    "GrokVisionAnalyze": GrokVisionAnalyze,
    "GrokVisionAnalyzeURL": GrokVisionAnalyzeURL,
    "LoadImageFromURL_DB": LoadImageFromURL,
    "NyquistNotch_DB": NyquistNotch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokVisionAnalyze": "Grok Vision Analyze (Image)",
    "GrokVisionAnalyzeURL": "Grok Vision Analyze (URL)",
    "LoadImageFromURL_DB": "Load Image From URL",
    "NyquistNotch_DB": "Nyquist Notch (2px VAE grid fix)",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
