# Computer Vision
from .huggingface_tasks.computer_vision.depth_estimation.node import LoadDepthModel, DepthEstimationInference
from .huggingface_tasks.computer_vision.image_classification.node import ImageClassificationPipeline
from .huggingface_tasks.computer_vision.image_segmentation.node import ImageSegmentationPipeline
from .huggingface_tasks.computer_vision.object_detection.node import ObjectDetectionPipeline
from .huggingface_tasks.computer_vision.image_to_image.node import ImageToImagePipeline
from .huggingface_tasks.computer_vision.image_feature_extraction.node import ImageFeatureExtractionPipeline
from .huggingface_tasks.computer_vision.zero_shot_image_classification.node import ZeroShotImageClassificationPipeline
from .huggingface_tasks.computer_vision.zero_shot_object_detection.node import ZeroShotObjectDetectionPipeline
from .huggingface_tasks.computer_vision.mask_generation.node import MaskGenerationPipeline
from .huggingface_tasks.computer_vision.video_classification.node import VideoClassificationPipeline


# NLP
from .huggingface_tasks.natural_language_processing.text_classification.node import TextClassificationPipeline
from .huggingface_tasks.natural_language_processing.text_generation.node import TextGenerationPipeline
from .huggingface_tasks.natural_language_processing.token_classification.node import TokenClassificationPipeline
from .huggingface_tasks.natural_language_processing.fill_mask.node import FillMaskPipeline
from .huggingface_tasks.natural_language_processing.question_answering.node import QuestionAnsweringPipeline
from .huggingface_tasks.natural_language_processing.zero_shot_classification.node import ZeroShotClassificationPipeline
from .huggingface_tasks.natural_language_processing.table_question_answering.node import TableQuestionAnsweringPipeline
from .huggingface_tasks.natural_language_processing.conversational.node import ConversationalPipeline
from .huggingface_tasks.natural_language_processing.sentence_similarity.node import SentenceSimilarityPipeline

# Audio
from .huggingface_tasks.audio.automatic_speech_recognition.node import AutomaticSpeechRecognitionPipeline
from .huggingface_tasks.audio.audio_classification.node import AudioClassificationPipeline
from .huggingface_tasks.audio.text2speech.node import TextToSpeechPipeline
from .huggingface_tasks.audio.zero_shot_audio_classification.node import ZeroShotAudioClassificationPipeline

# Utility
from .huggingface_tasks.utility.node import IntToString, StringToInt, FloatToString, StringToFloat

# Multimodal
from .huggingface_tasks.multimodal.feature_extraction.node import FeatureExtractionPipeline
from .huggingface_tasks.multimodal.image2text.node import ImageToTextPipeline
from .huggingface_tasks.multimodal.visual_question_answering.node import VisualQuestionAnsweringPipeline
from .huggingface_tasks.multimodal.document_question_answering.node import DocumentQuestionAnsweringPipeline
from .huggingface_tasks.multimodal.image_text_to_text.node import ImageTextToTextPipeline


NODE_CLASS_MAPPINGS = {
    # Computer Vision
    "LoadDepthModel": LoadDepthModel,
    "DepthEstimationInference": DepthEstimationInference,
    "ImageClassificationPipeline": ImageClassificationPipeline,
    "ImageSegmentationPipeline": ImageSegmentationPipeline,
    "ObjectDetectionPipeline": ObjectDetectionPipeline,
    "ImageToImagePipeline": ImageToImagePipeline,
    "ImageFeatureExtractionPipeline": ImageFeatureExtractionPipeline,
    "ZeroShotImageClassificationPipeline": ZeroShotImageClassificationPipeline,
    "ZeroShotObjectDetectionPipeline": ZeroShotObjectDetectionPipeline,
    "MaskGenerationPipeline": MaskGenerationPipeline,
    "VideoClassificationPipeline": VideoClassificationPipeline,
    # NLP
    "TextClassificationPipeline": TextClassificationPipeline,
    "TextGenerationPipeline": TextGenerationPipeline,
    "TokenClassificationPipeline": TokenClassificationPipeline,
    "FillMaskPipeline": FillMaskPipeline,
    "QuestionAnsweringPipeline": QuestionAnsweringPipeline,
    "ZeroShotClassificationPipeline": ZeroShotClassificationPipeline,
    "TableQuestionAnsweringPipeline": TableQuestionAnsweringPipeline,
    "ConversationalPipeline": ConversationalPipeline,
    "SentenceSimilarityPipeline": SentenceSimilarityPipeline,
    # Audio
    "AutomaticSpeechRecognitionPipeline": AutomaticSpeechRecognitionPipeline,
    "AudioClassificationPipeline": AudioClassificationPipeline,
    "TextToSpeechPipeline": TextToSpeechPipeline,
    "ZeroShotAudioClassificationPipeline": ZeroShotAudioClassificationPipeline,
    # Multimodal
    "FeatureExtractionPipeline": FeatureExtractionPipeline,
    "ImageToTextPipeline": ImageToTextPipeline,
    "VisualQuestionAnsweringPipeline": VisualQuestionAnsweringPipeline,
    "DocumentQuestionAnsweringPipeline": DocumentQuestionAnsweringPipeline,
    "ImageTextToTextPipeline": ImageTextToTextPipeline,
    # Utility
    "IntToString": IntToString,
    "StringToInt": StringToInt,
    "FloatToString": FloatToString,
    "StringToFloat": StringToFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Computer Vision
    "LoadDepthModel": "Load Depth Model",
    "DepthEstimationInference": "Depth Estimation",
    "ImageClassificationPipeline": "Image Classification",
    "ImageSegmentationPipeline": "Image Segmentation",
    "ObjectDetectionPipeline": "Object Detection",
    "ImageToImagePipeline": "Image to Image",
    "ImageFeatureExtractionPipeline": "Image Feature Extraction",
    "ZeroShotImageClassificationPipeline": "Zero-Shot Image Classification",
    "ZeroShotObjectDetectionPipeline": "Zero-Shot Object Detection",
    "MaskGenerationPipeline": "Mask Generation",
    "VideoClassificationPipeline": "Video Classification",
    # NLP
    "TextClassificationPipeline": "Text Classification",
    "TextGenerationPipeline": "Text Generation",
    "TokenClassificationPipeline": "Token Classification (NER)",
    "FillMaskPipeline": "Fill Mask",
    "QuestionAnsweringPipeline": "Question Answering",
    "ZeroShotClassificationPipeline": "Zero-Shot Classification",
    "TableQuestionAnsweringPipeline": "Table Question Answering",
    "ConversationalPipeline": "Conversational",
    "SentenceSimilarityPipeline": "Sentence Similarity",
    # Audio
    "AutomaticSpeechRecognitionPipeline": "Speech Recognition (ASR)",
    "AudioClassificationPipeline": "Audio Classification",
    "TextToSpeechPipeline": "Text to Speech",
    "ZeroShotAudioClassificationPipeline": "Zero-Shot Audio Classification",
    # Multimodal
    "FeatureExtractionPipeline": "Feature Extraction",
    "ImageToTextPipeline": "Image to Text",
    "VisualQuestionAnsweringPipeline": "Visual Question Answering",
    "DocumentQuestionAnsweringPipeline": "Document Question Answering",
    "ImageTextToTextPipeline": "Image-Text to Text",
    # Utility
    "IntToString": "Int to String",
    "StringToInt": "String to Int",
    "FloatToString": "Float to String",
    "StringToFloat": "String to Float",
}
