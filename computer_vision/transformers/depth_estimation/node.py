from transformers import pipeline
from PIL import Image
import numpy as np
import torch

depth_model_name_list = [
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
    "vinvino02/glpn-kitti",
    "Intel/dpt-beit-large-384",
    "Intel/dpt-beit-base-384",
    "Intel/dpt-beit-large-512",
    "LiheYoung/depth-anything-base-hf",
    "LiheYoung/depth-anything-small-hf",
    "LiheYoung/depth-anything-large-hf",
]

class DepthEstimationPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (depth_model_name_list, {"default": depth_model_name_list[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "depth_pipeline"

    CATEGORY = "ComputerVision/Transformers"

    def depth_pipeline(self, image, model_name):
        img = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        
        pipe = pipeline(task="depth-estimation", model=model_name)
        output = pipe(pil_image)
        tensor_image = torch.from_numpy(np.array(output["depth"]).astype(np.float32) / 255.0)[None,]
        return (tensor_image,)
