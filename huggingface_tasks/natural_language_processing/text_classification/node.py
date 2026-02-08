from transformers import pipeline as hf_pipeline

text_classification_model_list = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]


class TextClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": (text_classification_model_list, {"default": text_classification_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("label", "score",)
    FUNCTION = "run_classification"
    CATEGORY = "Transformers/NLP/TextClassification"

    def run_classification(self, text, model_name):
        pipe = hf_pipeline("text-classification", model=model_name)
        result = pipe(text)
        top = result[0]
        return (top["label"], top["score"],)
