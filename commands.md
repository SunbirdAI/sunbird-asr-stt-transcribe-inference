docker build --platform linux/amd64 --tag patrickcmd/worker-template:v1 .

docker push patrickcmd/worker-template:v1

MODEL_NAME: name of the model on huggingface
HF_TOKEN: (optional) your Hugging Face API token for private models.
CUDA version of the model
TOKENIZER_NAME: Tokenizer repository to use if any

Sample inputs to use for testing.
And any other setting that may be needed.

{
  "input": {
    "source_language": "lug",
    "target_language": "eng",
    "text": "Ogamba ki?"
  }
}


