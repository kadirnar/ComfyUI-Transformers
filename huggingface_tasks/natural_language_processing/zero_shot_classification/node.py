from transformers import pipeline as hf_pipeline
import json

class ZeroShotClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "candidate_labels": ("STRING", {"default": "positive, negative, neutral"}),
                "model_name": ("STRING", {"default": "facebook/bart-large-mnli"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "run_zero_shot"
    CATEGORY = "Transformers/NLP/ZeroShotClassification"

    def run_zero_shot(self, text, candidate_labels, model_name):
        labels = [l.strip() for l in candidate_labels.split(",")]
        pipe = hf_pipeline("zero-shot-classification", model=model_name, trust_remote_code=True)
        result = pipe(text, candidate_labels=labels)
        return (json.dumps(result, indent=2, default=str),)
