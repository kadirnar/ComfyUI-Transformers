from transformers import pipeline as hf_pipeline

class TextGenerationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "gpt2"}),
                "max_new_tokens": ("INT", {"default": 50, "min": 1, "max": 2048}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "run_generation"
    CATEGORY = "Transformers/NLP/TextGeneration"

    def run_generation(self, prompt, model_name, max_new_tokens, temperature):
        pipe = hf_pipeline("text-generation", model=model_name, trust_remote_code=True)
        result = pipe(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return (result[0]["generated_text"],)
