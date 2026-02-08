from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np

class DocumentQuestionAnsweringPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": "impira/layoutlm-document-qa"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("answer", "score",)
    FUNCTION = "run_doc_qa"
    CATEGORY = "Transformers/Multimodal/DocumentQA"

    def run_doc_qa(self, image, question, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("document-question-answering", model=model_name, trust_remote_code=True)
        result = pipe(pil_image, question)
        top = result[0]
        return (top["answer"], top.get("score", 0.0),)
