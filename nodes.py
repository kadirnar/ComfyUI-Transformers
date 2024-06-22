from .huggingface_tasks.computer_vision.depth_estimation.node import LoadDepthModel, DepthEstimationInference
from .huggingface_tasks.computer_vision.image_classification.node import ImageClassificationPipeline
from .huggingface_tasks.computer_vision.image_segmentation.node import ImageSegmentationPipeline
from .huggingface_tasks.computer_vision.object_detection.node import ObjectDetectionPipeline


NODE_CLASS_MAPPINGS = {
	"LoadDepthModel": LoadDepthModel,
	"DepthEstimationInference": DepthEstimationInference,
	"ImageClassificationPipeline": ImageClassificationPipeline,
	"ImageSegmentationPipeline": ImageSegmentationPipeline,
	"ObjectDetectionPipeline": ObjectDetectionPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"LoadDepthModel": "LoadDepthModel",
	"DepthEstimationInference": "DepthEstimation",
	"ImageClassificationPipeline": "ImageClassification",
	"ImageSegmentationPipeline": "ImageSegmentation",
	"ObjectDetectionPipeline": "ObjectDetection",
}
