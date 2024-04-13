""" Example handler file. """

import os
import sys
import time

import runpod
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from config import sb_lm_config  # noqa F401
from utils import KenLM, get_audio_file, load_model_and_processor, transcribe

load_dotenv()


current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.


def main(target_lang, adapter, audio_file):
    model_id = "facebook/mms-1b-all"
    target_lang = target_lang
    adapter = adapter
    audio_file = audio_file
    ngram_type = "eng_5gram"  # Specify the desired ngram type (e.g., "5gram", "3gram", "mixed_5gram", "mixed_3gram")

    lm_file_name = f"{target_lang}_{ngram_type}.bin"
    lm_file_subfolder = "language_model"

    try:
        lm_file = hf_hub_download(
            repo_id="Sunbird/sunbird-mms",
            filename=lm_file_name,
            subfolder=lm_file_subfolder,
        )
    except Exception as e:
        print(f"Error downloading language model file: {e}")
        return

    model, processor = load_model_and_processor(model_id, target_lang, adapter)
    kenlm = KenLM(processor.tokenizer, lm_file)

    transcription_with_lm = transcribe(audio_file, model, processor, kenlm)
    # transcription_without_lm = transcribe(audio_file, model, processor)

    return transcription_with_lm  # transcription_without_lm


def handler(job):
    """Handler function that will be used to process jobs."""
    try:
        job_input = job["input"]

        target_lang = job_input.get("target_lang", "lug")
        adapter = job_input.get("adapter", "lug")
        audio_file = get_audio_file(job_input.get("audio_file"))

        start_time = time.time()

        transcription_with_lm = main(target_lang, adapter, audio_file)
        response = {"audio_transcription": transcription_with_lm[0]}
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"Audio transcription execution time: {execution_time:.4f} seconds / {execution_time / 60:.4f} minutes"
        )
    except Exception as e:
        response = {"Error": str(e)}

    return response


runpod.serverless.start({"handler": handler})
