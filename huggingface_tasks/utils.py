import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline
import json

_pipeline_cache = {}


def get_pipeline(task, model_name, **kwargs):
    """Cached pipeline loader - avoids reloading models on every call."""
    key = (task, model_name, frozenset(kwargs.items()))
    if key not in _pipeline_cache:
        _pipeline_cache[key] = hf_pipeline(task, model=model_name, trust_remote_code=True, **kwargs)
    return _pipeline_cache[key]


def comfyui_to_pil(image_tensor):
    """ComfyUI IMAGE tensor (B,H,W,C) [0,1] -> PIL Image (first frame)."""
    img = 255.0 * image_tensor[0].cpu().numpy()
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def pil_to_comfyui(pil_image):
    """PIL Image -> ComfyUI IMAGE tensor (1,H,W,C) [0,1]."""
    arr = np.array(pil_image).astype(np.float32) / 255.0
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return torch.from_numpy(arr)[None,]


def format_output(data):
    """Convert pipeline output to formatted JSON string."""
    return json.dumps(data, indent=2, default=str)
