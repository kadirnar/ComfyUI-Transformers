from transformers import pipeline as hf_pipeline
import json

class TableQuestionAnsweringPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"default": "", "multiline": False}),
                "table_json": ("STRING", {"default": '{"Name": ["Alice", "Bob"], "Age": ["25", "30"]}', "multiline": True}),
                "model_name": ("STRING", {"default": "google/tapas-base-finetuned-wtq"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer",)
    FUNCTION = "run_table_qa"
    CATEGORY = "Transformers/NLP/TableQuestionAnswering"

    def run_table_qa(self, question, table_json, model_name):
        table = json.loads(table_json)
        pipe = hf_pipeline("table-question-answering", model=model_name, trust_remote_code=True)
        result = pipe(table=table, query=question)
        return (result["answer"],)
