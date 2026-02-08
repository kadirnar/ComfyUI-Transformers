from transformers import pipeline as hf_pipeline

text2text_generation_model_list = [
    "google/flan-t5-small",
    "google/flan-t5-base",
]


class Text2TextGenerationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": (text2text_generation_model_list, {"default": text2text_generation_model_list[0]}),
                "max_new_tokens": ("INT", {"default": 50, "min": 1, "max": 2048}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "run_text2text"
    CATEGORY = "Transformers/NLP/Text2TextGeneration"

    def run_text2text(self, text, model_name, max_new_tokens):
        pipe = hf_pipeline("text2text-generation", model=model_name)
        result = pipe(text, max_new_tokens=max_new_tokens)
        return (result[0]["generated_text"],)
