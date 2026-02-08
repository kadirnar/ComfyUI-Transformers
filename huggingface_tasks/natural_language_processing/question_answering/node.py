from transformers import pipeline as hf_pipeline

question_answering_model_list = [
    "distilbert-base-cased-distilled-squad",
    "deepset/roberta-base-squad2",
]


class QuestionAnsweringPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"default": "", "multiline": False}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "model_name": (question_answering_model_list, {"default": question_answering_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("answer", "score",)
    FUNCTION = "run_qa"
    CATEGORY = "Transformers/NLP/QuestionAnswering"

    def run_qa(self, question, context, model_name):
        pipe = hf_pipeline("question-answering", model=model_name)
        result = pipe(question=question, context=context)
        return (result["answer"], result["score"],)
