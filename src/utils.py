import base64
import json
import os

import torch
from dotenv import load_dotenv
from google.cloud import storage
from huggingface_hub import hf_hub_download
from pyctcdecode import build_ctcdecoder
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lang_config = {"ach": "Sunbird/sunbird-mms", "lug": "Sunbird/sunbird-mms"}


def setup_model(model_id: str, language: str):
    """
    Load Wav2Vec2 model for the specified language.

    Args:
        model_id (str): Identifier for the Wav2Vec2 model.
        language (str): Language code.

    Returns:
        model: Loaded Wav2Vec2 model.
        tokenizer: Model tokenizer.
        processor: Processor for the model.
        feature_extractor: Feature extractor for the model.
    """
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
    tokenizer.set_target_lang(language)
    if language == "eng":
        model.load_adapter(language)
    else:
        model.load_adapter(f"{language}+eng")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    return model, tokenizer, processor, feature_extractor


def setup_decoder(language: str, tokenizer, feature_extractor):
    """
    Setup CTC decoder with language model.

    Args:
        language (str): Language code.
        tokenizer: Model tokenizer.
        feature_extractor: Feature extractor for the model.

    Returns:
        decoder: CTC decoder.
    """
    if language in ["ach", "lug"]:
        lm_file_name = f"{language}_eng_3gram.bin"
        lm_file_subfolder = "language_model"
        lm_file = hf_hub_download(
            repo_id=lang_config[language],
            filename=lm_file_name,
            subfolder=lm_file_subfolder,
        )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {
        k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
    }
    if language in ["ach", "lug"]:
        decoder = build_ctcdecoder(
            labels=list(sorted_vocab_dict.keys()), kenlm_model_path=lm_file
        )
    else:
        decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()))
    return decoder


def setup_pipeline(model, language, tokenizer, feature_extractor, processor, decoder):
    """
    Setup ASR pipeline.

    Args:
        model: Loaded Wav2Vec2 model.
        language (str): Language code.
        tokenizer: Model tokenizer.
        feature_extractor: Feature extractor for the model.
        processor: Processor for the model.
        decoder: CTC decoder.

    Returns:
        pipe: ASR pipeline.
    """
    if language in ["ach", "lug"]:
        processor_with_lm = Wav2Vec2ProcessorWithLM(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            decoder=decoder,
        )
        feature_extractor._set_processor_class("Wav2Vec2ProcessorWithLM")
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor_with_lm.tokenizer,
            feature_extractor=processor_with_lm.feature_extractor,
            decoder=processor_with_lm.decoder,
            device=device,
            chunk_length_s=5,
            stride_length_s=(1, 2),
        )
    else:
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            decoder=decoder,
            device=device,
            chunk_length_s=5,
            stride_length_s=(1, 2),
        )

    return pipe


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


def transcribe_audio(pipe, audio_file: str):
    """
    Transcribe audio file using the given pipeline.

    Args:
        pipe: ASR pipeline.
        audio_file (str): Path to the audio file.

    Returns:
        str: Transcription of the audio file.
    """
    return pipe(audio_file)


if __name__ == "__main__":
    model_id = "Sunbird/sunbird-mms"
    language = "ach"
    model, tokenizer, processor, feature_extractor = setup_model(model_id, language)
    decoder = setup_decoder(language, tokenizer, feature_extractor)
    pipe = setup_pipeline(model, tokenizer, feature_extractor, processor, decoder)

    audio_files = [
        "./content/MEGA 12.2.mp3",
        # "./content/Lutino weng pwonye - Dul 1 - Introduction - Including Radio Maria.mp3",
    ]

    for audio_file in audio_files:
        if os.path.exists(audio_file):
            transcription = transcribe_audio(pipe, audio_file)
            print(f"Transcription for {os.path.basename(audio_file)}: {transcription}")
        else:
            print(f"File {audio_file} does not exist.")
