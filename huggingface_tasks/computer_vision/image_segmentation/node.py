from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import torch

class ImageSegmentationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "category_name": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": "mattmdjaga/segformer_b2_clothes"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "image_segmentation_pipeline"
    CATEGORY = "Transformers/ImageSegmentation"

    def image_segmentation_pipeline(self, image, category_name, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline(task="image-segmentation", model=model_name, trust_remote_code=True)
        result = pipe(pil_image)

        for item in result:
            if item["label"] == category_name:
                mask = item["mask"]
                arr = np.array(mask).astype(np.float32) / 255.0
                if len(arr.shape) == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                return (torch.from_numpy(arr)[None,],)

        h, w = pil_image.size[1], pil_image.size[0]
        empty = torch.zeros(1, h, w, 3)
        return (empty,)
