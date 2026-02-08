from transformers import pipeline as hf_pipeline

class TextClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "distilbert-base-uncased-finetuned-sst-2-english"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("label", "score",)
    FUNCTION = "run_classification"
    CATEGORY = "Transformers/NLP/TextClassification"

    def run_classification(self, text, model_name):
        pipe = hf_pipeline("text-classification", model=model_name, trust_remote_code=True)
        result = pipe(text)
        top = result[0]
        return (top["label"], top["score"],)
