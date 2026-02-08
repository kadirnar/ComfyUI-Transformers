from transformers import pipeline as hf_pipeline
from PIL import Image, ImageDraw
import numpy as np
import torch
import json

class ObjectDetectionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ("STRING", {"default": "facebook/detr-resnet-50"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("annotated_image", "detections_json",)
    FUNCTION = "object_detection_pipeline"
    CATEGORY = "Transformers/ObjectDetection"

    def object_detection_pipeline(self, image, model_name, threshold):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("object-detection", model=model_name, trust_remote_code=True)
        results = pipe(pil_image, threshold=threshold)

        draw = ImageDraw.Draw(pil_image)
        for det in results:
            box = det["box"]
            x, y, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            draw.rectangle((x, y, x2, y2), outline="red", width=2)
            label_text = f"{det['label']} ({det['score']:.2f})"
            draw.text((x, y), label_text, fill="white")

        arr = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(arr)[None,]
        detections_json = json.dumps(results, indent=2, default=str)

        return (tensor_image, detections_json,)
