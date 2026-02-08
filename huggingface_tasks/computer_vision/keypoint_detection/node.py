from transformers import pipeline as hf_pipeline
from PIL import Image, ImageDraw
import numpy as np
import torch
import json

keypoint_detection_model_list = [
    "magic-leap-community/superpoint",
]


class KeypointDetectionPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (keypoint_detection_model_list, {"default": keypoint_detection_model_list[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("annotated_image", "keypoints_json",)
    FUNCTION = "run_keypoint_detection"
    CATEGORY = "Transformers/ComputerVision/KeypointDetection"

    def run_keypoint_detection(self, image, model_name):
        img = 255.0 * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        pipe = hf_pipeline("keypoint-detection", model=model_name)
        result = pipe(pil_image)

        draw = ImageDraw.Draw(pil_image)
        keypoints_data = []
        if result and len(result) > 0:
            for detection in result:
                keypoints = detection.get("keypoints", [])
                for kp in keypoints:
                    x, y = kp[0], kp[1]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")
                    keypoints_data.append({"x": float(x), "y": float(y)})

        arr = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(arr)[None,]
        return (tensor_image, json.dumps(keypoints_data, indent=2),)
