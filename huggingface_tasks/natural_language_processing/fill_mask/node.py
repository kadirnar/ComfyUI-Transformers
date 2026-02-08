from transformers import pipeline as hf_pipeline
import json

class FillMaskPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "The capital of France is [MASK].", "multiline": True}),
                "model_name": ("STRING", {"default": "bert-base-uncased"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("predictions_json",)
    FUNCTION = "run_fill_mask"
    CATEGORY = "Transformers/NLP/FillMask"

    def run_fill_mask(self, text, model_name, top_k):
        pipe = hf_pipeline("fill-mask", model=model_name, trust_remote_code=True)
        result = pipe(text, top_k=top_k)
        return (json.dumps(result, indent=2, default=str),)
