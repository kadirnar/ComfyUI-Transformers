from transformers import pipeline as hf_pipeline
import json

class VideoClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": "MCG-NJU/videomae-base-finetuned-kinetics"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "run_video_classification"
    CATEGORY = "Transformers/VideoClassification"

    def run_video_classification(self, video_path, model_name, top_k):
        pipe = hf_pipeline("video-classification", model=model_name, trust_remote_code=True)
        result = pipe(video_path, top_k=top_k)
        return (json.dumps(result, indent=2, default=str),)
