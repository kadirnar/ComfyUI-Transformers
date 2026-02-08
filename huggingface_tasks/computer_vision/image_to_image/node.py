from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import torch

image_to_image_model_list = [
    "caidas/swin2SR-classical-sr-x2-64",
    "caidas/swin2SR-lightweight-x2-64",
]


class ImageToImagePipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (image_to_image_model_list, {"default": image_to_image_model_list[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run_image_to_image"
    CATEGORY = "Transformers/ComputerVision/ImageToImage"

    def run_image_to_image(self, image, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("image-to-image", model=model_name)
        result = pipe(pil_image)

        arr = np.array(result).astype(np.float32) / 255.0
        if len(arr.shape) == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return (torch.from_numpy(arr)[None,],)
