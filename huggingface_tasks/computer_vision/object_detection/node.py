from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw
import torch


object_detection_model_name_list = [
    "mattmdjaga/segformer_b2_clothes",
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
]

class ObjectDetectionPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "category_name": ("STRING",),
                "model_name": (object_detection_model_name_list, {"default": object_detection_model_name_list[0]}),
                "threshold": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_segmentation_pipeline"

    CATEGORY = "ComputerVision/Transformers"


def object_detection_pipeline(image_path, model_name="facebook/detr-resnet-50", threshold=0.5):
    """
    Detects objects in an image using a pre-trained object detection model, draws bounding boxes around them,
    and prints the labels and scores of detected objects.

    Args:
    image_path (str): The path to the input image.
    model_name (str): The name of the pre-trained object detection model.
    threshold (float): The confidence threshold for object detection.

    Returns:
    PIL.Image.Image: The image with bounding boxes drawn around detected objects.
    """

    # Load the image
    image = Image.open(image_path)

    # Load the image processor and model
    image_processor = AutoImageProcessor.from_pretrained(model_name, revision="no_timm")
    model = AutoModelForObjectDetection.from_pretrained(model_name, revision="no_timm")

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    # Draw bounding boxes, labels, and scores on the image
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        label_text = f"{model.config.id2label[label.item()]} ({score:.2f})"
        draw.text((x, y), label_text, fill="white")

    return (image,)
