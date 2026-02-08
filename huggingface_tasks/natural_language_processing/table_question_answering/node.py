from transformers import pipeline as hf_pipeline
import json

table_qa_model_list = [
    "google/tapas-base-finetuned-wtq",
    "google/tapas-large-finetuned-wtq",
]


class TableQuestionAnsweringPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"default": "", "multiline": False}),
                "table_json": ("STRING", {"default": '{"Name": ["Alice", "Bob"], "Age": ["25", "30"]}', "multiline": True}),
                "model_name": (table_qa_model_list, {"default": table_qa_model_list[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer",)
    FUNCTION = "run_table_qa"
    CATEGORY = "Transformers/NLP/TableQuestionAnswering"

    def run_table_qa(self, question, table_json, model_name):
        table = json.loads(table_json)
        pipe = hf_pipeline("table-question-answering", model=model_name)
        result = pipe(table=table, query=question)
        return (result["answer"],)
