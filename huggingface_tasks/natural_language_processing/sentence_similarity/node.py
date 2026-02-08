from transformers import pipeline as hf_pipeline
import numpy as np

class SentenceSimilarityPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"default": "", "multiline": True}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "sentence-transformers/all-MiniLM-L6-v2"}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity_score",)
    FUNCTION = "run_similarity"
    CATEGORY = "Transformers/NLP/SentenceSimilarity"

    def run_similarity(self, text1, text2, model_name):
        pipe = hf_pipeline("feature-extraction", model=model_name, trust_remote_code=True)
        emb1 = np.array(pipe(text1)[0]).mean(axis=0)
        emb2 = np.array(pipe(text2)[0]).mean(axis=0)

        cosine_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return (cosine_sim,)
