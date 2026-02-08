from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import torch
import comfy.model_management as mm
from comfy.utils import ProgressBar
import torchvision.transforms.functional as F

class LoadDepthModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "Intel/dpt-hybrid-midas"}),
            },
        }

    RETURN_TYPES = ("DEPTH_MODEL", "IMAGE_PROCESSOR")
    FUNCTION = "load_depth_model"

    CATEGORY = "Transformers/DepthEstimation"

    def load_depth_model(self, model_name):
        model = AutoModelForDepthEstimation.from_pretrained(model_name, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        return (model, image_processor)

class DepthEstimationInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("DEPTH_MODEL",),
                "processor": ("IMAGE_PROCESSOR",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "depth_inference"

    CATEGORY = "Transformers/DepthEstimation"

    def depth_inference(self, image, model, processor):
        # Convert ComfyUI image to PIL image
        img = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        
        # Prepare the image for the model
        inputs = processor(images=pil_image, return_tensors="pt")
        pbar = ProgressBar(len(inputs["pixel_values"]))

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Convert depth map to tensor in ComfyUI format
        tensor_image = prediction.squeeze().cpu().numpy()
        tensor_image = tensor_image.astype(np.float32) / tensor_image.max()
        tensor_image = torch.from_numpy(tensor_image)[None,]

        return (tensor_image,)


NODE_CLASS_MAPPINGS = {
    "LoadDepthModel": LoadDepthModel,
    "DepthEstimationInference": DepthEstimationInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDepthModel": "Load Depth Model",
    "DepthEstimationInference": "Depth Estimation Inference",
}
