from transformers import pipeline as hf_pipeline
import json

audio_classification_model_list = [
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
]


class AudioClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "model_name": (audio_classification_model_list, {"default": audio_classification_model_list[0]}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "run_audio_classification"
    CATEGORY = "Transformers/Audio/AudioClassification"

    def run_audio_classification(self, audio_path, model_name, top_k):
        pipe = hf_pipeline("audio-classification", model=model_name)
        result = pipe(audio_path, top_k=top_k)
        return (json.dumps(result, indent=2, default=str),)
