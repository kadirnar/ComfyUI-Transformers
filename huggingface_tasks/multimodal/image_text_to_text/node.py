from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np

class ImageTextToTextPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "Salesforce/blip2-opt-2.7b"}),
                "max_new_tokens": ("INT", {"default": 50, "min": 1, "max": 512}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "run_image_text_to_text"
    CATEGORY = "Transformers/Multimodal/ImageTextToText"

    def run_image_text_to_text(self, image, prompt, model_name, max_new_tokens):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("image-text-to-text", model=model_name, trust_remote_code=True)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        result = pipe(pil_image, text=messages, max_new_tokens=max_new_tokens)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                return (result[0].get("generated_text", str(result[0])),)
        return (str(result),)
