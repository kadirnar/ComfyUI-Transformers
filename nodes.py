from .huggingface_tasks.computer_vision.depth_estimation.node import DepthEstimationPipeline
from .huggingface_tasks.computer_vision.image_classification.node import ImageClassificationPipeline
from .huggingface_tasks.computer_vision.image_segmentation.node import ImageSegmentationPipeline
from .huggingface_tasks.computer_vision.object_detection.node import ObjectDetectionPipeline
from .huggingface_tasks.multimodal.image2text.node import FlorenceMultiModelPipeline


NODE_CLASS_MAPPINGS = {
	"DepthEstimationPipeline": DepthEstimationPipeline,
	"ImageClassificationPipeline": ImageClassificationPipeline,
	"ImageSegmentationPipeline": ImageSegmentationPipeline,
	"ObjectDetectionPipeline": ObjectDetectionPipeline,
	"MultiModelPipeline": FlorenceMultiModelPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"DepthEstimationPipeline": "DepthEstimation",
	"ImageClassificationPipeline": "ImageClassification",
	"ImageSegmentationPipeline": "ImageSegmentation",
	"ObjectDetectionPipeline": "ObjectDetection",
	"MultiModelPipeline": "FlorenceMultiModelPipeline",
}
