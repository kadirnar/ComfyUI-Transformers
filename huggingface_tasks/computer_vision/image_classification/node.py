from transformers import pipeline as hf_pipeline
import numpy as np
from PIL import Image

class ImageClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ("STRING", {"default": "microsoft/resnet-50"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("label", "score",)
    FUNCTION = "classify_image"
    CATEGORY = "Transformers/ImageClassification"

    def classify_image(self, image, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("image-classification", model=model_name, trust_remote_code=True)
        result = pipe(pil_image)

        top = result[0]
        return (top["label"], top["score"],)
