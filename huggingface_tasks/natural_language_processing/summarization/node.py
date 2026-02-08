from transformers import pipeline as hf_pipeline

summarization_model_list = [
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
]


class SummarizationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": (summarization_model_list, {"default": summarization_model_list[0]}),
                "max_length": ("INT", {"default": 130, "min": 10, "max": 1024}),
                "min_length": ("INT", {"default": 30, "min": 1, "max": 512}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary_text",)
    FUNCTION = "run_summarization"
    CATEGORY = "Transformers/NLP/Summarization"

    def run_summarization(self, text, model_name, max_length, min_length):
        pipe = hf_pipeline("summarization", model=model_name)
        result = pipe(text, max_length=max_length, min_length=min_length)
        return (result[0]["summary_text"],)
