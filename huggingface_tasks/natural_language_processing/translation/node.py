from transformers import pipeline as hf_pipeline

translation_model_list = [
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-en-fr",
]


class TranslationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": (translation_model_list, {"default": translation_model_list[0]}),
                "max_length": ("INT", {"default": 512, "min": 10, "max": 2048}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translation_text",)
    FUNCTION = "run_translation"
    CATEGORY = "Transformers/NLP/Translation"

    def run_translation(self, text, model_name, max_length):
        pipe = hf_pipeline("translation", model=model_name)
        result = pipe(text, max_length=max_length)
        return (result[0]["translation_text"],)
