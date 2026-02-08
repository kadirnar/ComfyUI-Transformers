from transformers import pipeline as hf_pipeline
import numpy as np

sentence_similarity_model_list = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


class SentenceSimilarityPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"default": "", "multiline": True}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "model_name": (sentence_similarity_model_list, {"default": sentence_similarity_model_list[0]}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity_score",)
    FUNCTION = "run_similarity"
    CATEGORY = "Transformers/NLP/SentenceSimilarity"

    def run_similarity(self, text1, text2, model_name):
        pipe = hf_pipeline("feature-extraction", model=model_name)
        emb1 = np.array(pipe(text1)[0]).mean(axis=0)
        emb2 = np.array(pipe(text2)[0]).mean(axis=0)

        cosine_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return (cosine_sim,)
