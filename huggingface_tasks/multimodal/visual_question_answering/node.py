from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np
import json

class VisualQuestionAnsweringPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": "dandelin/vilt-b32-finetuned-vqa"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answers_json",)
    FUNCTION = "run_vqa"
    CATEGORY = "Transformers/Multimodal/VisualQA"

    def run_vqa(self, image, question, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("visual-question-answering", model=model_name, trust_remote_code=True)
        result = pipe(pil_image, question)
        return (json.dumps(result, indent=2, default=str),)
