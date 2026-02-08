from transformers import pipeline as hf_pipeline
import json

video_classification_model_list = [
    "MCG-NJU/videomae-base-finetuned-kinetics",
    "MCG-NJU/videomae-small-finetuned-kinetics",
]


class VideoClassificationPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "model_name": (video_classification_model_list, {"default": video_classification_model_list[0]}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results_json",)
    FUNCTION = "run_video_classification"
    CATEGORY = "Transformers/ComputerVision/VideoClassification"

    def run_video_classification(self, video_path, model_name, top_k):
        pipe = hf_pipeline("video-classification", model=model_name)
        result = pipe(video_path, top_k=top_k)
        return (json.dumps(result, indent=2, default=str),)
