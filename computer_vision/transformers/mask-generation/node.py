from transformers import pipeline
from PIL import Image

mask_generation_model_name_list = [
    "mattmdjaga/segformer_b2_clothes",
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
]

class MaskGenerationPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (mask_generation_model_name_list, {"default": mask_generation_model_name_list[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_generation_pipeline"

    CATEGORY = "ComputerVision/Transformers"


def mask_generation_pipeline(image_path: str, model_name: str) -> Image.Image:
    pipe = pipeline(task="image-to-image", model=model_name)
    result = pipe(image_path)

    return (result,)
