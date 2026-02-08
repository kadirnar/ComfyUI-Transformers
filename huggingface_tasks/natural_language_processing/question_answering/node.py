from transformers import pipeline as hf_pipeline

class QuestionAnsweringPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"default": "", "multiline": False}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "distilbert-base-cased-distilled-squad"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("answer", "score",)
    FUNCTION = "run_qa"
    CATEGORY = "Transformers/NLP/QuestionAnswering"

    def run_qa(self, question, context, model_name):
        pipe = hf_pipeline("question-answering", model=model_name, trust_remote_code=True)
        result = pipe(question=question, context=context)
        return (result["answer"], result["score"],)
