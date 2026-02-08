from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import json

class ZeroShotImageClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "candidate_labels": ("STRING", {"default": "cat, dog, bird"}),
                "model_name": ("STRING", {"default": "openai/clip-vit-base-patch32"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "run_zero_shot_classification"
    CATEGORY = "Transformers/ZeroShotImageClassification"

    def run_zero_shot_classification(self, image, candidate_labels, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        labels = [l.strip() for l in candidate_labels.split(",")]
        pipe = hf_pipeline("zero-shot-image-classification", model=model_name, trust_remote_code=True)
        result = pipe(pil_image, candidate_labels=labels)
        return (json.dumps(result, indent=2, default=str),)
