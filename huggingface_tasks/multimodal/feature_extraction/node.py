from transformers import pipeline as hf_pipeline
import numpy as np
import json

class FeatureExtractionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "sentence-transformers/all-MiniLM-L6-v2"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("features_json",)
    FUNCTION = "run_feature_extraction"
    CATEGORY = "Transformers/Multimodal/FeatureExtraction"

    def run_feature_extraction(self, text, model_name):
        pipe = hf_pipeline("feature-extraction", model=model_name, trust_remote_code=True)
        result = pipe(text)
        features = np.array(result[0])
        output = {"shape": list(features.shape), "features_preview": features[0][:10].tolist()}
        return (json.dumps(output, indent=2),)
