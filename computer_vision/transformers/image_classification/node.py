from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
from typing import Tuple

image_classification_model_name_list = [
    "microsoft/resnet-50"
]

class ImageClassificationPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (image_classification_model_name_list, {"default": image_classification_model_name_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING","FLOAT",)
    FUNCTION = "classify_pipeline"

    CATEGORY = "ComputerVision/Transformers"

    def classify_image(image: Image.Image, model_name: str) -> Tuple[str, float]:
        """
        Classifies a given image using the specified model and returns the name and score of the predicted label.

        Args:
        image (PIL.Image.Image): The image to be classified.
        model_name (str): The name of the model to be used for classification.

        Returns:
        Tuple[str, float]: The name of the predicted label and its corresponding score.
        """
        
        # Load the specified model and corresponding image processor
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = ResNetForImageClassification.from_pretrained(model_name)

        # Process the image and prepare it for the model
        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        # Predict the class with the highest score
        predicted_label_id = logits.argmax(-1).item()
        predicted_label_name = model.config.id2label[predicted_label_id]
        predicted_score = torch.softmax(logits, -1)[0, predicted_label_id].item()

        return predicted_label_name, predicted_score

