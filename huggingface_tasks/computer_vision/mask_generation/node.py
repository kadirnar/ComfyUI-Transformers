from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import torch
import json

mask_generation_model_list = [
    "facebook/sam-vit-base",
    "facebook/sam-vit-large",
]


class MaskGenerationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (mask_generation_model_list, {"default": mask_generation_model_list[0]}),
                "points_per_batch": ("INT", {"default": 64, "min": 1, "max": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("overlay_image", "masks_json",)
    FUNCTION = "run_mask_generation"
    CATEGORY = "Transformers/ComputerVision/MaskGeneration"

    def run_mask_generation(self, image, model_name, points_per_batch):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("mask-generation", model=model_name, points_per_batch=points_per_batch)
        result = pipe(pil_image)

        overlay = np.array(pil_image).astype(np.float32)
        masks_info = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, mask in enumerate(result["masks"][:10]):
            mask_arr = np.array(mask)
            color = colors[i % len(colors)]
            for c in range(3):
                overlay[:, :, c] = np.where(mask_arr, overlay[:, :, c] * 0.5 + color[c] * 0.5, overlay[:, :, c])
            masks_info.append({"mask_index": i, "score": float(result["scores"][i]) if "scores" in result else None})

        overlay = np.clip(overlay, 0, 255).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(overlay)[None,]
        return (tensor_image, json.dumps(masks_info, indent=2),)
