from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import json

image_feature_extraction_model_list = [
    "google/vit-base-patch16-224",
    "facebook/dinov2-small",
]


class ImageFeatureExtractionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (image_feature_extraction_model_list, {"default": image_feature_extraction_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("features_json",)
    FUNCTION = "run_feature_extraction"
    CATEGORY = "Transformers/ComputerVision/ImageFeatureExtraction"

    def run_feature_extraction(self, image, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("image-feature-extraction", model=model_name)
        result = pipe(pil_image)

        features = np.array(result[0]).tolist()
        output = {"shape": list(np.array(result[0]).shape), "features_preview": features[:5] if isinstance(features[0], list) else features[:10]}
        return (json.dumps(output, indent=2),)
