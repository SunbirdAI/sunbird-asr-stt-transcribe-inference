""" Example handler file. """

import os
import sys
import time

import runpod
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import pipeline

from config import sb_lm_config  # noqa F401
from utils import (
    device,
    get_audio_file,
    load_kenlm_model,
    transcribe_with_kenlm,
    transcribe_without_kenlm,
)

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
    ngram_type = "eng_3gram"  # Specify the desired ngram type (e.g., "5gram", "3gram", "mixed_5gram", "mixed_3gram")
    chunk_length_s = 5  # Specify the desired chunk length in seconds
    stride_length_s = (
        1,
        2,
    )  # Specify the desired stride length in seconds (left, right)
    return_timestamps = (
        "word"  # Specify the desired timestamp format ("word" or "char")
    )
    lm_file_name = f"{target_lang}_{ngram_type}.bin"
    lm_file_subfolder = "language_model"

    asr_pipeline = pipeline(
        "automatic-speech-recognition", model=model_id, revision="main", device=device
    )
    asr_pipeline.tokenizer.set_target_lang(target_lang)
    asr_pipeline.model.load_adapter(adapter)

    try:
        lm_file = hf_hub_download(
            repo_id="Sunbird/sunbird-mms",
            filename=lm_file_name,
            subfolder=lm_file_subfolder,
        )
        kenlm_decoder = load_kenlm_model(lm_file, asr_pipeline.tokenizer)
        transcription_with_lm = transcribe_with_kenlm(
            audio_file,
            asr_pipeline,
            kenlm_decoder=kenlm_decoder,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            return_timestamps=return_timestamps,
        )
        return transcription_with_lm
    except Exception as e:
        print(f"Error downloading language model file: {e}")
        transcription_without_lm = transcribe_without_kenlm(
            audio_file,
            asr_pipeline,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            return_timestamps=return_timestamps,
        )
        return transcription_without_lm


def handler(job):
    """Handler function that will be used to process jobs."""
    try:
        job_input = job["input"]

        target_lang = job_input.get("target_lang", "lug")
        adapter = job_input.get("adapter", "lug")
        audio_file = get_audio_file(job_input.get("audio_file"))

        start_time = time.time()

        transcription = main(target_lang, adapter, audio_file)
        response = {"audio_transcription": transcription.get("text")}
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"Audio transcription execution time: {execution_time:.4f} seconds / {execution_time / 60:.4f} minutes"
        )
    except Exception as e:
        response = {"Error": str(e)}

    return response


runpod.serverless.start({"handler": handler})
