"""Nyquist Notch — removes the 2px repeating grid that the Qwen-Image (and, more
subtly, Wan-2.1) VAE leaves after decode.

The VAE stamps a pixel-alternating flicker into the image; if you sharpen or filter
afterwards it gets amplified into an ugly grid. This detects that 2px-period
component out to 7 taps and subtracts it. Wire it right after VAE Decode, before any
sharpening.

7-tap separable binomial notch:  b = [-1, 6, -15, 20, -15, 6, -1] / 64
out = (I - Hx)(I - Hy) * image

Ported to PyTorch from the author's GLSL node (identical math).
"""
import torch
import torch.nn.functional as F

_B = [-1.0, 6.0, -15.0, 20.0, -15.0, 6.0, -1.0]


class NyquistNotch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "0 = original, 1 = fully de-gridded."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "image/postprocessing"

    def apply(self, image, strength=1.0):
        orig_dtype = image.dtype
        x = image.movedim(-1, 1).float()  # [B,H,W,C] -> [B,C,H,W]
        c = x.shape[1]
        k = torch.tensor(_B, device=x.device, dtype=x.dtype) / 64.0
        kv = k.view(1, 1, 7, 1).repeat(c, 1, 1, 1)  # vertical, depthwise
        kh = k.view(1, 1, 1, 7).repeat(c, 1, 1, 1)  # horizontal, depthwise

        def conv_v(t):
            return F.conv2d(F.pad(t, (0, 0, 3, 3), mode="replicate"), kv, groups=c)

        def conv_h(t):
            return F.conv2d(F.pad(t, (3, 3, 0, 0), mode="replicate"), kh, groups=c)

        tmp = x - conv_v(x)  # (I - Hy)
        notched = (tmp - conv_h(tmp)).clamp_(0.0, 1.0)  # (I - Hx)(I - Hy)

        if strength < 1.0:
            notched = x * (1.0 - strength) + notched * strength

        return (notched.movedim(1, -1).to(orig_dtype),)
