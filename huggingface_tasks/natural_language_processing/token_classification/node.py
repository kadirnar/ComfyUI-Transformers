from transformers import pipeline as hf_pipeline
import json

token_classification_model_list = [
    "dslim/bert-base-NER",
    "Jean-Baptiste/camembert-ner",
]

aggregation_strategy_list = [
    "simple",
    "first",
    "average",
    "max",
    "none",
]


class TokenClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": (token_classification_model_list, {"default": token_classification_model_list[0]}),
                "aggregation_strategy": (aggregation_strategy_list, {"default": "simple"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("entities_json",)
    FUNCTION = "run_token_classification"
    CATEGORY = "Transformers/NLP/TokenClassification"

    def run_token_classification(self, text, model_name, aggregation_strategy):
        pipe = hf_pipeline("token-classification", model=model_name, aggregation_strategy=aggregation_strategy)
        result = pipe(text)
        return (json.dumps(result, indent=2, default=str),)
