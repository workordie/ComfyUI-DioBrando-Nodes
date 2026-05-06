# ComfyUI-DioBrando-Nodes

Two ComfyUI custom nodes I use day-to-day:

- **Grok Vision Analyze** — send an image (tensor or URL) to xAI's Grok vision API and get a text response back. Useful for auto-captioning, prompt generation from reference images, or quick visual QA.
- **Load Image From URL** — fetch a remote image directly into ComfyUI as `IMAGE` + `MASK` tensors. No download/upload step needed.

## Install

Clone into your ComfyUI custom_nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DanielBartolic/ComfyUI-DioBrando-Nodes
cd ComfyUI-DioBrando-Nodes
pip install -r requirements.txt   # only Pillow + numpy + torch, all standard ComfyUI deps
```

Restart ComfyUI. The nodes show up under the `DioBrando/` category.

## Grok Vision setup

Set your xAI API key as an environment variable before launching ComfyUI:

```bash
export XAI_API="xai-..."
```

Or paste it into the optional `api_key` input on the node (overrides env var).

### Models supported

```
grok-4.3-latest          ← default
grok-4.3
grok-4.20-multi-agent-0309
grok-4.20-0309-reasoning
grok-4.20-0309-non-reasoning
grok-4-1-fast-reasoning      ← fastest + cheapest
grok-4-1-fast-non-reasoning  ← fastest + cheapest
grok-4-vision
grok-3-vision-beta
grok-2-vision-1212
grok-vision-beta
```

## Nodes

| Node | Inputs | Outputs |
|---|---|---|
| **Grok Vision Analyze (Image)** | `IMAGE` tensor + prompt | `response`, `usage` |
| **Grok Vision Analyze (URL)** | image URL string + prompt | `response`, `usage` |
| **Load Image From URL** | URL string | `IMAGE`, `MASK`, `url` |

## Author

[@diobrando0](https://huggingface.co/diobrando0) on HuggingFace · [@workordie](https://civitai.com/user/workordie) on Civitai · [DanielBartolic](https://github.com/DanielBartolic) on GitHub.

## License

MIT — see [LICENSE](LICENSE).
