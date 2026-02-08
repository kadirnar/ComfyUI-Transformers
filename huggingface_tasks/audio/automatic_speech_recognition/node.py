from transformers import pipeline as hf_pipeline

class AutomaticSpeechRecognitionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": "openai/whisper-tiny"}),
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
        pipe = hf_pipeline("automatic-speech-recognition", model=model_name, trust_remote_code=True)
        result = pipe(audio_path, generate_kwargs=generate_kwargs)
        return (result["text"],)
