from transformers import pipeline as hf_pipeline
from PIL import Image, ImageDraw
import numpy as np
import torch
import json

class ZeroShotObjectDetectionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "candidate_labels": ("STRING", {"default": "cat, dog, person"}),
                "model_name": ("STRING", {"default": "google/owlvit-base-patch32"}),
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("annotated_image", "detections_json",)
    FUNCTION = "run_zero_shot_detection"
    CATEGORY = "Transformers/ZeroShotObjectDetection"

    def run_zero_shot_detection(self, image, candidate_labels, model_name, threshold):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        labels = [l.strip() for l in candidate_labels.split(",")]
        pipe = hf_pipeline("zero-shot-object-detection", model=model_name, trust_remote_code=True)
        results = pipe(pil_image, candidate_labels=labels, threshold=threshold)

        draw = ImageDraw.Draw(pil_image)
        for det in results:
            box = det["box"]
            x, y, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            draw.rectangle((x, y, x2, y2), outline="red", width=2)
            label_text = f"{det['label']} ({det['score']:.2f})"
            draw.text((x, y), label_text, fill="white")

        arr = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(arr)[None,]
        return (tensor_image, json.dumps(results, indent=2, default=str),)
