from transformers import pipeline as hf_pipeline
import json

class ConversationalPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_message": ("STRING", {"default": "", "multiline": True}),
                "history": ("STRING", {"default": "[]", "multiline": True}),
                "model_name": ("STRING", {"default": "microsoft/DialoGPT-medium"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "updated_history",)
    FUNCTION = "run_conversation"
    CATEGORY = "Transformers/NLP/Conversational"

    def run_conversation(self, user_message, history, model_name):
        history_list = json.loads(history) if history.strip() else []

        messages = []
        for turn in history_list:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_message})

        pipe = hf_pipeline("text-generation", model=model_name, trust_remote_code=True)
        result = pipe(messages, max_new_tokens=128, pad_token_id=pipe.tokenizer.eos_token_id)
        response = result[0]["generated_text"][-1]["content"]

        history_list.append({"user": user_message, "assistant": response})
        return (response, json.dumps(history_list, indent=2),)
