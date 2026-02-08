from transformers import pipeline as hf_pipeline
import scipy.io.wavfile
import numpy as np
import tempfile
import os

class TextToSpeechPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "suno/bark-small"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "run_tts"
    CATEGORY = "Transformers/Audio/TextToSpeech"

    def run_tts(self, text, model_name):
        pipe = hf_pipeline("text-to-speech", model=model_name, trust_remote_code=True)
        result = pipe(text)

        audio_data = result["audio"]
        sampling_rate = result["sampling_rate"]

        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)

        output_path = os.path.join(tempfile.gettempdir(), "comfyui_tts_output.wav")
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_data)
        return (output_path,)
