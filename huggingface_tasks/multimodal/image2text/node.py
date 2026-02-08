from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np

class ImageToTextPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ("STRING", {"default": "Salesforce/blip-image-captioning-base"}),
                "max_new_tokens": ("INT", {"default": 50, "min": 1, "max": 512}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "run_image2text"
    CATEGORY = "Transformers/Multimodal/ImageToText"

    def run_image2text(self, image, model_name, max_new_tokens):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("image-to-text", model=model_name, trust_remote_code=True)
        result = pipe(pil_image, max_new_tokens=max_new_tokens)
        return (result[0]["generated_text"],)
