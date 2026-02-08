from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import json

zero_shot_image_classification_model_list = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
]


class ZeroShotImageClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "candidate_labels": ("STRING", {"default": "cat, dog, bird"}),
                "model_name": (zero_shot_image_classification_model_list, {"default": zero_shot_image_classification_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "run_zero_shot_classification"
    CATEGORY = "Transformers/ComputerVision/ZeroShotImageClassification"

    def run_zero_shot_classification(self, image, candidate_labels, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        labels = [l.strip() for l in candidate_labels.split(",")]
        pipe = hf_pipeline("zero-shot-image-classification", model=model_name)
        result = pipe(pil_image, candidate_labels=labels)
        return (json.dumps(result, indent=2, default=str),)
