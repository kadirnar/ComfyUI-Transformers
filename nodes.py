from transformers import pipeline
from PIL import Image
import numpy as np
import torch

task_list = [
    "depth-estimation",
    "image-classification",
    "object-detection",
    "image-segmentation",
]

depth_model_name_list = [
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
    "vinvino02/glpn-kitti",
    "Intel/dpt-beit-large-384",
    "Intel/dpt-beit-base-384",
    "Intel/dpt-beit-large-512" 
]

class TransformersPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "task_type": (task_list, {"default": task_list[0]}),
                "model_name": (depth_model_name_list, {"default": depth_model_name_list[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "task_pipeline"

    CATEGORY = "HF_Vision"

    def task_pipeline(self, image, task_type, model_name):
        img = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        
        pipe = pipeline(task=task_type, model=model_name)
        output = pipe(pil_image)
        tensor_image = torch.from_numpy(np.array(output["depth"]).astype(np.float32) / 255.0)[None,]
        return (tensor_image,)

NODE_CLASS_MAPPINGS = {
	"TransformersPipeline": TransformersPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"TransformersPipeline": "Transformers Pipeline",

}
