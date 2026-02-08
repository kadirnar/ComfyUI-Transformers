from transformers import pipeline as hf_pipeline
import json

zero_shot_audio_classification_model_list = [
    "laion/clap-htsat-unfused",
    "laion/larger_clap_general",
]


class ZeroShotAudioClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "candidate_labels": ("STRING", {"default": "speech, music, noise"}),
                "model_name": (zero_shot_audio_classification_model_list, {"default": zero_shot_audio_classification_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "run_zero_shot_audio"
    CATEGORY = "Transformers/Audio/ZeroShotAudioClassification"

    def run_zero_shot_audio(self, audio_path, candidate_labels, model_name):
        labels = [l.strip() for l in candidate_labels.split(",")]
        pipe = hf_pipeline("zero-shot-audio-classification", model=model_name)
        result = pipe(audio_path, candidate_labels=labels)
        return (json.dumps(result, indent=2, default=str),)
