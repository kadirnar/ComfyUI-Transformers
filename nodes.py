from .huggingface_tasks.computer_vision.transformers.depth_estimation.node import DepthEstimationPipeline
from .huggingface_tasks.computer_vision.transformers.image_classification.node import ImageClassificationPipeline
from .huggingface_tasks.computer_vision.transformers.image_segmentation.node import ImageSegmentationPipeline
from .huggingface_tasks.computer_vision.transformers.object_detection.node import ObjectDetectionPipeline


NODE_CLASS_MAPPINGS = {
	"DepthEstimationPipeline": DepthEstimationPipeline,
	"ImageClassificationPipeline": ImageClassificationPipeline,
	"ImageSegmentationPipeline": ImageSegmentationPipeline,
	"ObjectDetectionPipeline": ObjectDetectionPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"DepthEstimationPipeline": "DepthEstimation",
	"ImageClassificationPipeline": "ImageClassification",
	"ImageSegmentationPipeline": "ImageSegmentation",
	"ObjectDetectionPipeline": "ObjectDetection",
}
