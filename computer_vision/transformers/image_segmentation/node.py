from transformers import pipeline
from PIL import Image

image_segmentation_model_name_list = [
    "mattmdjaga/segformer_b2_clothes",
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
]

class ImageSegmentationPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "category_name": ("STRING",),
                "model_name": (image_segmentation_model_name_list, {"default": image_segmentation_model_name_list[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_segmentation_pipeline"

    CATEGORY = "ComputerVision/Transformers"


def image_segmentation_pipeline(image_path: str, category_name: str, model_name: str) -> Image.Image:
    """
    Given an image, a category name, and a model name, extracts and returns the mask for the specified category.

    Args:
    image_path (str): The path to the image file.
    category_name (str): The name of the category for which the mask is to be extracted.
    model_name (str): The name of the model to be used for image segmentation.

    Returns:
    PIL.Image.Image: The mask of the specified category, if found. Otherwise, returns None.

    Example:
    >>> image_path = "path/to/image.jpg"
    >>> category_name = "car"
    >>> model_name = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    >>> mask = image_segmentation_pipeline(image_path, category_name, model_name)
    """

    # Create the pipeline with the specified model
    pipe = pipeline(task="image-segmentation", model=model_name)

    # Process the image
    result = pipe(image_path)

    # Search for the mask of the specified category
    for item in result:
        if item['label'] == category_name:
            return (item['mask'],)

    # If the category is not found, return None
    return (None,)
