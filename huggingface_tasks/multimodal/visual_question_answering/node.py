from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import json

vqa_model_list = [
    "dandelin/vilt-b32-finetuned-vqa",
    "Salesforce/blip-vqa-base",
]


class VisualQuestionAnsweringPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"default": "", "multiline": False}),
                "model_name": (vqa_model_list, {"default": vqa_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answers_json",)
    FUNCTION = "run_vqa"
    CATEGORY = "Transformers/Multimodal/VisualQA"

    def run_vqa(self, image, question, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("visual-question-answering", model=model_name)
        result = pipe(pil_image, question)
        return (json.dumps(result, indent=2, default=str),)
