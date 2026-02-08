from transformers import pipeline as hf_pipeline

asr_model_list = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
]


class AutomaticSpeechRecognitionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "model_name": (asr_model_list, {"default": asr_model_list[0]}),
                "language": ("STRING", {"default": "english"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcribed_text",)
    FUNCTION = "run_asr"
    CATEGORY = "Transformers/Audio/SpeechRecognition"

    def run_asr(self, audio_path, model_name, language):
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        pipe = hf_pipeline("automatic-speech-recognition", model=model_name)
        result = pipe(audio_path, generate_kwargs=generate_kwargs)
        return (result["text"],)
