from transformers import pipeline as hf_pipeline
import numpy as np
import json

feature_extraction_model_list = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


class FeatureExtractionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": (feature_extraction_model_list, {"default": feature_extraction_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("features_json",)
    FUNCTION = "run_feature_extraction"
    CATEGORY = "Transformers/Multimodal/FeatureExtraction"

    def run_feature_extraction(self, text, model_name):
        pipe = hf_pipeline("feature-extraction", model=model_name)
        result = pipe(text)
        features = np.array(result[0])
        output = {"shape": list(features.shape), "features_preview": features[0][:10].tolist()}
        return (json.dumps(output, indent=2),)
