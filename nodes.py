from transformers import pipeline
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

task_list = [
    "depth-estimation",
    "image-classification",
    "object-detection",
    "image-segmentation",
]

depth_model_name = [
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large"
]

class TransformersPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "task_type": ("LIST", task_list),
                "model_name": ("LIST", depth_model_name),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "task_pipeline"

    CATEGORY = "HF_Vision"

    def task_pipeline(self, image, task_type, model_name):
        img = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        
        logging.info(f"Task: {task_type}", f"Model: {model_name}")
        pipe = pipeline(task=task_type, model=model_name)
        output = pipe(pil_image)
        
        output_image = output["predicted_depth"]
        tensor_image = output["depth"]
        
        return (tensor_image, output_image,)

NODE_CLASS_MAPPINGS = {
	"TransformersPipeline": TransformersPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"TransformersPipeline": "Transformers Pipeline",

}