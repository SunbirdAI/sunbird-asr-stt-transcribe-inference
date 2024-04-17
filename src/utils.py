import base64
import json
import os
from multiprocessing import Pool

import librosa
import torch
from dotenv import load_dotenv
from google.cloud import storage
from pyctcdecode import build_ctcdecoder

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KenLM:
    def __init__(self, tokenizer, model_name, num_workers=8, beam_width=128):
        self.num_workers = num_workers
        self.beam_width = beam_width
        vocab_dict = tokenizer.get_vocab()
        self.vocabulary = [
            x[0] for x in sorted(vocab_dict.items(), key=lambda x: x[1], reverse=False)
        ]

        self.decoder = build_ctcdecoder(self.vocabulary, model_name)

    @staticmethod
    def lm_postprocess(text):
        return " ".join([x if len(x) > 1 else "" for x in text.split()]).strip()

    def decode(self, logits):
        probs = logits.cpu().numpy()
        with Pool(self.num_workers) as pool:
            text = self.decoder.decode_batch(pool, probs)
        text = [KenLM.lm_postprocess(x) for x in text]
        return text


def decode_gcp_credentials():
    encoded_credentials = os.getenv("GCP_CREDENTIALS")
    decoded_bytes = base64.b64decode(encoded_credentials)
    decoded_string = decoded_bytes.decode("utf-8")
    decoded_json = json.loads(decoded_string)

    decoded_cred_json_file_path = "credentials.json"
    with open(decoded_cred_json_file_path, "w") as f:
        json.dump(decoded_json, f, indent=4)

    return decoded_cred_json_file_path


def download_audio_file(bucket_name, blob_name, folder_path):
    try:
        if os.path.exists("./credentials.json"):
            credentials_json = "./credentials.json"
        else:
            credentials_json = decode_gcp_credentials()
        storage_client = storage.Client.from_service_account_json(credentials_json)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        filename = blob_name.split("/")[-1]
        file_path = os.path.join(folder_path, filename)

        # Download the file to the specified folder
        blob.download_to_filename(file_path)

        return file_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_audio_file(audio_file):
    if os.path.exists(audio_file):
        audio_file = audio_file
    else:
        bucket_name = os.getenv("AUDIO_CONTENT_BUCKET_NAME")
        blob_name = audio_file
        folder_path = "./content"
        audio_file = download_audio_file(bucket_name, blob_name, folder_path)

    return audio_file


def load_kenlm_model(lm_file, tokenizer):
    kenlm_model = KenLM(tokenizer, lm_file)
    print("Loaded KenLM model from:", lm_file)
    return kenlm_model


def transcribe_with_kenlm(
    audio_file,
    asr_pipeline,
    kenlm_decoder,
    chunk_length_s=None,
    stride_length_s=None,
    return_timestamps=None,
):
    audio_samples = librosa.load(audio_file, sr=16000, mono=True)[0]

    print("kenlm decoder type", asr_pipeline.type)

    return asr_pipeline(
        audio_samples,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps=return_timestamps,
    )


def transcribe_without_kenlm(
    audio_file,
    asr_pipeline,
    chunk_length_s=None,
    stride_length_s=None,
    return_timestamps=None,
):
    audio_samples = librosa.load(audio_file, sr=16000, mono=True)[0]

    # print(asr_pipeline.type)
    print("no lmhead decoder type", asr_pipeline.type)

    return asr_pipeline(
        audio_samples,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps=return_timestamps,
    )
