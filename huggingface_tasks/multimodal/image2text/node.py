import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

depth_model_name_list = [
    "microsoft/Florence-2-large"
]

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)


class FlorenceMultiModelPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (depth_model_name_list, {"default": depth_model_name_list[0]}),
            },
            "optional": {
                "task_type": (["Caption", "Detailed Caption", "More Detailed Caption",
                               "Object Detection", "Dense Region Caption",
                               "Region Proposal", "Caption to Phrase Grounding",
                               "Referring Expression Segmentation", "Region to Segmentation",
                               "Open Vocabulary Detection", "Region to Category",
                               "Region to Description", "OCR", "OCR with Region", "None"],),
                "prompt": ("STRING",),

            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "multimodel_pipeline"

    CATEGORY = "ComputerVision/Transformers/Image2Text"

    def multimodel_pipeline(self, image: torch.tensor, text_input: str, task_type: str, max_new_tokens: int = 1024):
        """

        Args:
            image:
            text_input:
            task_type:
            max_new_tokens

        Returns:

        """

        tensor_image = image.permute(0, 3, 1, 2)
        inputs = processor(text=text_input, images=tensor_image, return_tensors="pt")
        task_prompt = task_parser(task_prompt=task_type)
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(generated_text, task=prompt,
                                                          image_size=(image.width, image.height))

        image = plot_bbox(image, parsed_answer)

        return (image, )


def task_parser(task_prompt: str):
    """

    Args:
        task_prompt:

    Returns:

    """
    if task_prompt == 'Caption':
        task_prompt = '<CAPTION>'
    elif task_prompt == 'Detailed Caption':
        task_prompt = '<DETAILED_CAPTION>'
    elif task_prompt == 'More Detailed Caption':
        task_prompt = '<MORE_DETAILED_CAPTION>'
    elif task_prompt == 'Object Detection':
        task_prompt = '<OD>'
    elif task_prompt == 'Dense Region Caption':
        task_prompt = '<DENSE_REGION_CAPTION>'
    elif task_prompt == 'Region Proposal':
        task_prompt = '<REGION_PROPOSAL>'
    elif task_prompt == 'Caption to Phrase Grounding':
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    elif task_prompt == 'Referring Expression Segmentation':
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
    elif task_prompt == 'Region to Segmentation':
        task_prompt = '<REGION_TO_SEGMENTATION>'
    elif task_prompt == 'Open Vocabulary Detection':
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
    elif task_prompt == 'Region to Category':
        task_prompt = '<REGION_TO_CATEGORY>'
    elif task_prompt == 'Region to Description':
        task_prompt = '<REGION_TO_DESCRIPTION>'
    elif task_prompt == 'OCR':
        task_prompt = '<OCR>'
    elif task_prompt == 'OCR with Region':
        task_prompt = '<OCR_WITH_REGION>'
    elif task_prompt == 'None':
        return None

    return task_prompt


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_bbox(image, data):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    return fig